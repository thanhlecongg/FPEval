# FPEval: A Holistic Evaluation Framework for Code Generation in Functional Programming

[![Build and test](https://github.com/thanhlecongg/FormalBench/actions/workflows/build_and_test.yml/badge.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/badge/Release-0.1.0-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-v1.0-yellow.svg)](https://huggingface.co/FPEvalRepoPublic)


This repository contains evaluation infrastructure for FPEval, an holistic evaluation framework for code generation in functional programming. FPEval includes its FPBench benchmark, evaluation infrastructures and wrappers for calling LLMs. If you found this repository to be useful, please cite our research paper:

```
@article{fpeval,
  title={Perish or Flourish? A Holistic Evaluation of Large Language Models for Code Generation in Functional Programming},
  author={Lang, Nguyet-Anh H and Lang, Eric and Le-Cong, Thanh and Le, Bach and Huynh, Quyet-Thang},
  journal={arXiv preprint arXiv:2601.02060},
  year={2026}
}
```

## Prerequisites and Compatibility:

### Supported Models
- OpenAI series (GPT3.5, GPT-4o)
- DeepSeek (DeepSeekV3, DeepSeekR1) (on plan)
- Claude series (Sonnet, Haiku, Opus) (on plan)
- CodeQwen series (on plan)
- CodeLLama series (on plan)
- Mixtral series (on plan)

### Programming Languages
This project is built based on the following programming languages:
- Python 3.10.16
- Haskell (ghc) 9.4.8
- OCaml 4.08.1
- Java 11.0.26
- sbt 1.10.11
- Scala 2.11.12

### OS
This project is built and well-tested on Unbuntu 20.04.4. 


Make sure to use correct environment before getting started. 

---

## Dataset
The dataset is available at [HuggingFace](https://huggingface.co/LLMFP).

## Installation

Choose one of the following options to install the project:

### Option 1: Manual Installation

Set up environments for each of the programming languages.
Then use the following command to install our Python enviroment:
```bash
# Build the Docker image
conda env create -f environment.yml
```

### Option 2: Using Docker
Either build the Docker image or pull the image from Docker Hub.

#### 2.1: Get the Docker Image
```bash
# Build the Docker image
docker build -t llm4fp .

# OR
# Pull image from Docker Hub
docker pull erlang123/llm4fp:v4
```

#### 2.2 Set environment variables
This project environment variables (mainly API keys) to be set in order to access various LLMs.
Please create a file named `.env`, using `.env.example` as a template.
The `.env` file is given as an argument when running the docker containers as seen in `2.3`.

#### 2.3 Run the Docker image
```bash
# If you built the Docker image
docker run --platform linux/amd64 -it --rm \
  -v $(pwd):/home/conda-user/project \
  --env-file .env \
  llm4fp

# If you pulled the image from Docker Hub
docker run --platform linux/amd64 -it --rm \
  -v $(pwd):/home/conda-user/project \
  --env-file .env \
  erlang123/llm4fp:v4

# Run detached
docker run --platform linux/amd64 -d --rm \
  -v $(pwd):/home/conda-user/project \
  --env-file .env \
  erlang123/llm4fp:v4 tail -f /dev/null
```
Note that the container does not contain application code - it uses a volume to mount the local directory.

### Verification
TODO

---

## Usage


### Basic Commands
There is the python script to run the code:
1. `run_dataset.py`: This script is used to generate code for a dataset. Used to benchmark the code.

#### 1.1: Run dataset
```bash
python3 run_dataset.py --token [HUGGINGFACE_TOKEN] \
                --repo_id [HUGGINFACE_REPO_ID] \ # default: "LLMFP/LeetCodeProble"
                --language [LANGUAGE] \ # default: haskell, options: haskell, ocaml, scala
                --workflow [WORKFLOW] \ # default: basic, options: basic, self-repair, instruction-repair, code-execution
                --model_name [MODEL_NAME] \ # default: gpt-3.5-turbo, options: gpt-3.5-turbo, gpt-4o, gpt-5
                --output_path [OUTPUT_PATH], 
```

Example usage: 
```
python3 run_dataset.py --token my_hf_token
```
This will automatically download the dataset from the HuggingFace repository (which will be cached locally in ./huggingface_cache) and generate code for all the files in the dataset.

- file_path is path to the input data with above structure
- workflow is workflow of LLM-based agent using Langgraph. Please see `workflows/` for diagrams of these workflows. Currently, we support
    - `basic`
    - `self-repair`
    - `instruction-repair`
    - `code-execution`
- language is the target programming language
- model_name is base LLM for each workflow. Currently, we support,
    - gpt-3.5-turbo, gpt-4o, gpt-5 (OpenAI)
- output_path is path to a folder for storing results

### Pull Request

#### Readiness Checklist
- [ ] If documentation is needed for this change, has that been included in this pull request
- [ ] If any additional Python libraries are required, please update `environment.yml`

### Version Note
- Not release yet
