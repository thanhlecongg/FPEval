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

# Configure logging with a single logger instance
logging.basicConfig(
    level=logging.INFO,
    # %(asctime)s - 
    format='%(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a single logger instance for the entire application
logger = logging.getLogger('FPEval')

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