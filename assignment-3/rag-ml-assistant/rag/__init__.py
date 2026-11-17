"""RAG Machine Learning Knowledge Assistant."""
__version__ = "1.0.0"

try:  # optional: load API keys from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed -> use real environment variables
    pass
