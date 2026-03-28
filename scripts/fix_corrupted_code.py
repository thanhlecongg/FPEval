"""
fix_corrupted_code.py

Uses vLLM offline inference to fix corrupted output JSON files — no HTTP server needed.

Corruption pattern:
  The AI response at messages[idx=2] uses a plain code fence (` ``` `) without
  a language tag, e.g.:
      ```
      rangeAddQueries ...
      ```
  Expected format:
      ```haskell
      rangeAddQueries ...
      ```
  This causes downstream parsers to fail when extracting code_traces.

Fix strategy (batch, offline):
  1. Scan all JSON files and collect corrupted ones.
  2. Reconstruct the original system + human turns for each corrupted file.
  3. Feed ALL conversations to vLLM in one llm.chat() call (continuous batching).
  4. Patch each file with its fixed response and updated code_traces.
"""

import os
import re
import json
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (override via env vars or CLI args)
# ---------------------------------------------------------------------------
DEFAULT_MODEL   = os.getenv("MODEL_NAME", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
DEFAULT_OUTPUT  = os.getenv("OUTPUT_PATH", "./output")

SUPPORTED_LANGUAGES = ["ocaml", "haskell", "scala", "java"]

# vLLM sampling defaults
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS  = 2048


# ---------------------------------------------------------------------------
# Data container for a single corrupted file
# ---------------------------------------------------------------------------
@dataclass
class CorruptedEntry:
    json_path: Path
    output_root: Path      # root used to compute relative paths for mirroring
    language: str
    data: dict
    messages: list[dict]   # system + human turns ready for llm.chat()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------
def get_ai_message(data: dict) -> dict | None:
    """Return the messages entry with idx=2 and role='ai', or None."""
    for msg in data.get("messages", []):
        if msg.get("idx") == 2 and msg.get("role") == "ai":
            return msg
    return None


def is_corrupted(data: dict) -> bool:
    """
    A file is corrupted when the AI response (idx=2) opens with a plain
    ``` fence instead of ```<language>.

    BAD:   "```\\nlet foo ..."
    GOOD:  "```ocaml\\nlet foo ..."
    """
    ai_msg = get_ai_message(data)
    # print(ai_msg)
    if ai_msg is None:
        return False
    response: str = ai_msg.get("response", "")
    return response.startswith("```\n") or response.startswith("```\r\n")


# ---------------------------------------------------------------------------
# Prompt / patch helpers
# ---------------------------------------------------------------------------
FIX_FORMAT_SYSTEM = (
    "You are a code formatter. Your only task is to fix the opening code fence "
    "of the provided code block by adding the correct language tag. "
    "Do NOT change, rewrite, reorder, or add any code. "
    "Output only the corrected code block, nothing else."
)


def build_messages_for_llm(data: dict, language: str) -> list[dict]:
    """
    Build a two-turn conversation that asks the LLM to fix only the code fence
    format (add the language tag) without modifying any code.
    """
    ai_msg = get_ai_message(data)
    corrupted_response = ai_msg.get("response", "") if ai_msg else ""
    return [
        {"role": "system", "content": FIX_FORMAT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"The following code block is missing the language tag in its opening fence. "
                f"Fix it by changing the opening ``` to ```{language}. "
                f"Do not modify anything else.\n\n{corrupted_response}"
            ),
        },
    ]


def extract_code_from_response(response: str, language: str = "") -> str:
    """
    Extract the inner code from a fenced block.

    Double-checks the fix: if the LLM correctly added the language tag,
    use a language-specific pattern; otherwise fall back to a generic one.
    Returns the inner code, or the whole response if no fence is found.
    """
    if language and f"```{language}" in response:
        pattern = rf"```{language}(.*?)```"
    else:
        pattern = r"```(?:\w+)?(.*?)```"

    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: strip first/last fence lines manually
    lines = response.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def patch_output(data: dict, model_name: str, fixed_response: str,
                 prompt_tokens: int, completion_tokens: int,
                 language: str = "") -> dict:
    """Overwrite the corrupted AI message and regenerate code_traces."""
    metadata = {
        "token_usage": {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
        "model_name": model_name,
        "system_fingerprint": None,
        "finish_reason": "stop",
        "logprobs": None,
    }
    for msg in data["messages"]:
        if msg.get("idx") == 2 and msg.get("role") == "ai":
            msg["response"] = fixed_response
            msg["response_metadata"] = metadata
            break

    data["code_traces"] = [extract_code_from_response(fixed_response, language)]
    return data


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
SEP = "─" * 72

def _truncate(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"  ... ({len(lines) - max_lines} more lines)")
    return "\n".join(kept)


def find_language_dirs(output_root: Path, language: str) -> list[Path]:
    """
    Find all directories named <language> anywhere under output_root.
    This handles any number of intermediate levels (e.g. output/basics/haskell/).
    """
    return sorted(p for p in output_root.rglob(language) if p.is_dir())


def debug_print_scan(output_root: Path, languages: list[str]) -> None:
    """
    Print a recursive folder tree for every language dir found anywhere under
    output_root. Works at any nesting depth.
    """
    def _print_tree(directory: Path, prefix: str = "") -> None:
        dirs = sorted(c for c in directory.iterdir() if c.is_dir())
        n_json_here = len(list(directory.glob("*.json")))
        if n_json_here:
            print(f"{prefix}  ({n_json_here} json files)")
        for i, child in enumerate(dirs):
            branch = "└──" if i == len(dirs) - 1 else "├──"
            n_json = len(list(child.rglob("*.json")))
            print(f"{prefix}{branch} {child.name}/  [{n_json} json total]")
            extension = "    " if i == len(dirs) - 1 else "│   "
            _print_tree(child, prefix + extension)

    print(f"\n{SEP}")
    print(f"  SCAN PLAN  —  root: {output_root}")
    print(SEP)
    for language in languages:
        lang_dirs = find_language_dirs(output_root, language)
        if not lang_dirs:
            print(f"  [{language}]  (no directory found anywhere under root, skipping)")
            continue
        for lang_dir in lang_dirs:
            n_total = len(list(lang_dir.rglob("*.json")))
            rel = lang_dir.relative_to(output_root)
            print(f"  [{language}]  .../{rel}  [{n_total} json total]")
            _print_tree(lang_dir, prefix="    ")
    print(SEP + "\n")


def debug_print_entry_before(entry: CorruptedEntry) -> None:
    """Print the corrupted code snippet for a single entry."""
    ai_msg = get_ai_message(entry.data)
    corrupted_response = ai_msg.get("response", "") if ai_msg else ""
    print(f"\n{SEP}")
    print(f"  FILE     : {entry.json_path}")
    print(f"  LANGUAGE : {entry.language}")
    print(f"  BEFORE (corrupted response):")
    print(SEP)
    print(_truncate(corrupted_response))
    print(SEP)


def debug_print_entry_after(entry: CorruptedEntry, fixed_response: str) -> None:
    """Print the fixed response and the extracted code_traces result."""
    extracted = extract_code_from_response(fixed_response, entry.language)
    print(f"  AFTER (fixed response):")
    print(SEP)
    print(_truncate(fixed_response))
    print(SEP)
    print(f"  EXTRACTED code_traces[0]:")
    print(SEP)
    print(_truncate(extracted))
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Scan phase — collect all corrupted entries
# ---------------------------------------------------------------------------
def model_dir_of(json_path: Path) -> str:
    """
    Return the immediate parent directory of the json file — the model directory name.
    Works regardless of nesting depth.
    e.g. output/haskell/Qwen/Qwen2.5-Coder-3B/foo.json  →  "Qwen2.5-Coder-3B"
    e.g. output/haskell/Qwen2.5-Coder-3B/foo.json        →  "Qwen2.5-Coder-3B"
    """
    return json_path.parent.name


def collect_corrupted(output_root: Path, languages: list[str],
                      allowed_models: list[str] | None = None,
                      debug: bool = False) -> list[CorruptedEntry]:
    entries: list[CorruptedEntry] = []
    skipped_models: set[str] = set()

    if debug:
        debug_print_scan(output_root, languages)

    for language in languages:
        lang_dirs = find_language_dirs(output_root, language)
        if not lang_dirs:
            logger.warning(f"No directory named '{language}' found under {output_root}")
            continue

        for lang_dir in lang_dirs:
            logger.info(f"Scanning {lang_dir}")
            for json_path in sorted(lang_dir.rglob("*.json")):
                model_dir = model_dir_of(json_path)

                if allowed_models is not None and model_dir not in allowed_models:
                    skipped_models.add(model_dir)
                    continue

                try:
                    data = load_json(json_path)
                    if is_corrupted(data):
                        messages = build_messages_for_llm(data, language)
                        entries.append(CorruptedEntry(
                            json_path=json_path,
                            output_root=output_root,
                            language=language,
                            data=data,
                            messages=messages,
                        ))
                except Exception as e:
                    logger.error(f"Error reading {json_path}: {e}")

    if skipped_models:
        for m in sorted(skipped_models):
            logger.info(f"  [SKIPPED model] {m}")

    logger.info(f"Found {len(entries)} corrupted file(s).")
    return entries


# ---------------------------------------------------------------------------
# Batch inference with vLLM
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 32


def _resolve_save_path(entry: CorruptedEntry, fixed_output_root: Path | None) -> Path:
    """
    If fixed_output_root is set, mirror the input path under it.
    e.g. input:  /working/output/haskell/Qwen/Model/foo.json
         output: /working/output_fixed/haskell/Qwen/Model/foo.json
    Otherwise overwrite the original file.
    """
    if fixed_output_root is None:
        return entry.json_path
    rel = entry.json_path.relative_to(entry.output_root)
    return fixed_output_root / rel


def _process_outputs(batch_entries: list[CorruptedEntry], outputs,
                     model_name: str, fixed_output_root: Path | None,
                     debug: bool) -> None:
    """
    Save fixed files for one completed batch.
    When debug=True, print before+after for every entry in the batch.
    """
    for entry, output in zip(batch_entries, outputs):
        try:
            choice = output.outputs[0]
            fixed_response = choice.text

            prompt_tokens     = len(output.prompt_token_ids)
            completion_tokens = len(choice.token_ids)

            if debug:
                debug_print_entry_before(entry)
                debug_print_entry_after(entry, fixed_response)

            updated_data = patch_output(
                entry.data, model_name, fixed_response,
                prompt_tokens, completion_tokens,
                language=entry.language,
            )
            save_path = _resolve_save_path(entry, fixed_output_root)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(updated_data, save_path)
            logger.info(f"  [FIXED] {save_path}")
        except Exception as e:
            logger.error(f"  [ERROR] {entry.json_path}: {e}")


def run_batch(entries: list[CorruptedEntry], model_name: str,
              temperature: float, max_tokens: int,
              batch_size: int = DEFAULT_BATCH_SIZE,
              fixed_output_root: Path | None = None,
              debug: bool = False) -> None:
    """
    Load the model once, then process entries in chunks of `batch_size`,
    saving results after each chunk.
    If fixed_output_root is set, saves fixed files there mirroring the input
    structure instead of overwriting originals.
    """
    if fixed_output_root:
        logger.info(f"Fixed output root: {fixed_output_root}")
    else:
        logger.info("Fixed output root: (overwriting originals)")

    logger.info(f"Loading model: {model_name}")
    llm = LLM(model=model_name, tensor_parallel_size=4)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # chat_template_kwargs={"enable_thinking": False},
    )

    total     = len(entries)
    n_batches = (total + batch_size - 1) // batch_size
    logger.info(f"Total entries: {total}  |  batch size: {batch_size}  |  batches: {n_batches}")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)
        batch_entries = entries[start:end]

        logger.info(f"Batch {batch_idx + 1}/{n_batches}  ({start}–{end - 1})")
        conversations = [e.messages for e in batch_entries]
        outputs = llm.chat(conversations, sampling_params=sampling_params)
        _process_outputs(batch_entries, outputs, model_name, fixed_output_root, debug)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix corrupted LLM output JSON files using vLLM offline inference."
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL,  help="HuggingFace model ID or local path")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT, help="Root output directory")
    parser.add_argument(
        "--languages", nargs="+", default=SUPPORTED_LANGUAGES,
        choices=SUPPORTED_LANGUAGES, help="Languages to process",
    )
    parser.add_argument("--models_file", default="./list_model_to_run.txt",
                        help="Path to a text file listing model directory names to process, one per line")
    parser.add_argument("--fixed_output_path", default=None,
                        help="Directory to write fixed files (mirrors input structure). "
                             "If omitted, originals are overwritten.")
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens",  type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--dry_run", action="store_true",
                        help="Scan and log corrupted files without calling LLM")
    parser.add_argument("--debug", action="store_true",
                        help="Print folder tree, corrupted code (before) and fixed code (after)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load allowed models from file
    models_file = Path(args.models_file)
    if models_file.exists():
        allowed_models = [
            line.strip() for line in models_file.read_text().splitlines()
            if line.strip()
        ]
        logger.info(f"Loaded {len(allowed_models)} model(s) from {models_file}")
    else:
        logger.warning(f"Models file not found: {models_file} — processing all models")
        allowed_models = None

    logger.info(f"Fix model  : {args.model}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Languages  : {args.languages}")
    logger.info(f"Models file: {args.models_file}")
    logger.info(f"Dry run    : {args.dry_run}")
    logger.info(f"Debug      : {args.debug}")

    output_root = Path(args.output_path)
    entries = collect_corrupted(output_root, args.languages,
                                allowed_models=allowed_models, debug=args.debug)

    if not entries:
        logger.info("Nothing to fix.")
        return

    if args.dry_run:
        for e in entries:
            logger.info(f"  [DRY-RUN] {e.json_path}  ({len(e.messages)} turns)")
        return

    fixed_output_root = Path(args.fixed_output_path) if args.fixed_output_path else None
    run_batch(entries, args.model, args.temperature, args.max_tokens,
              batch_size=args.batch_size, fixed_output_root=fixed_output_root,
              debug=args.debug)
    logger.info("Done.")


if __name__ == "__main__":
    main()
