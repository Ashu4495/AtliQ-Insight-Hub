import chromadb
import os

persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    print(f"Collections found: {[c.name for c in collections]}")
    for c in collections:
        print(f"Collection '{c.name}' has {c.count()} items.")
else:
    print(f"Directory {persist_dir} does not exist.")
