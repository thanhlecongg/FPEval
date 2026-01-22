import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
import os

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"DEBUG: Current API Key ends with: ...{api_key}")
    else:
        print("DEBUG: NO API KEY FOUND in environment variables!")
main()
if "OPENAI_ORGANIZATION" in os.environ:
    del os.environ["OPENAI_ORGANIZATION"]
    print("DEBUG: Deleted OPENAI_ORGANIZATION env var to avoid conflicts.")