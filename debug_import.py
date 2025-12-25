
import traceback
import sys

print("Attempting import...")
try:
    from sentence_transformers import SentenceTransformer
    print("Import successful")
except Exception:
    traceback.print_exc()
