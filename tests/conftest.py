"""Force offline settings for every test, whatever .env says or however pytest was invoked.

Without this, a developer's .env (PROVIDER=ollama, EMBED_MODEL=nomic-embed-text) could leak in
and the tests would silently call a local LLM, or fail when it is not running.
"""
import os

os.environ["EMBED_MODEL"] = "local"
os.environ["PROVIDER"] = "openai"  # never selects Ollama; tests use a stub LLM and local hashing embeddings

from src import config  # noqa: E402
from src.llm import get_embeddings  # noqa: E402

config.EMBED_MODEL = "local"
config.PROVIDER = "openai"
get_embeddings.cache_clear()
