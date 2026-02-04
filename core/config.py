import os
import logging
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# LANGUAGE = os.getenv("LANGUAGE")
# if LANGUAGE:
#     logging.debug(f"Language: {LANGUAGE}")
# else:
#     logging.warning("NO LANGUAGE FOUND in environment variables!")

# Configure logging with both console and file handlers
logger = logging.getLogger('FPEval')
logger.setLevel(logging.DEBUG)

# Prevent duplicate logs if handlers already exist
if logger.handlers:
    logger.handlers.clear()

# Create formatters
formatter = logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler - logs to file
file_handler = logging.FileHandler('FPEval_stats.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler - logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress httpx INFO logs (HTTP request logs)
logging.getLogger('httpx').setLevel(logging.WARNING)

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.debug(f"Current API Key ends with: ...{api_key}")
    else:
        logger.warning("NO API KEY FOUND in environment variables!")
main()
if "OPENAI_ORGANIZATION" in os.environ:
    del os.environ["OPENAI_ORGANIZATION"]
    logger.debug("Deleted OPENAI_ORGANIZATION env var to avoid conflicts.")