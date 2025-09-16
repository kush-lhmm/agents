import os

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
ASSETS_DIR = os.path.join(ROOT, "assets")
CSV_PATH = os.path.join(ASSETS_DIR, "sample-file.csv")

CHROMA_DIR = os.path.join(ROOT, "chroma_data")
VSTORE_DIR = os.path.join(ROOT, "vector_store_data")

DOCS_INDEX_JSON = os.path.join(VSTORE_DIR, "docs_index.json")