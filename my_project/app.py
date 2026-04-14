import os
import json
import requests
import msgpack
from sentence_transformers import SentenceTransformer

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "ai_knowledge"
DIM = 384  # all-MiniLM-L6-v2 output dimension

model = SentenceTransformer("all-MiniLM-L6-v2")

import re

base = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base, "data.txt"), "r") as f:
    raw = f.read()
lines = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]


def create_index():
    res = requests.post(f"{ENDEE_URL}/api/v1/index/create", json={
        "index_name": INDEX_NAME,
        "dim": DIM,
        "space_type": "cosine"
    })
    print("Index:", res.text)


def index_documents():
    vectors = model.encode(lines).tolist()
    documents = [
        {"id": str(i + 1), "vector": vectors[i], "meta": json.dumps({"text": lines[i]})}
        for i in range(len(lines))
    ]
    res = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
        json=documents,
        headers={"Content-Type": "application/json"}
    )
    print("Indexed:", res.text)


def search(query, top_k=1):
    query_vector = model.encode([query])[0].tolist()
    res = requests.post(
        f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search",
        json={"vector": query_vector, "k": top_k}
    )
    return msgpack.unpackb(res.content, raw=False)


if __name__ == "__main__":
    create_index()
    index_documents()

    print("\nSemantic Search ready. Type a query (or 'quit' to exit):\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() == "quit":
            break
        results = search(query)
        print("\nBest Match:")
        for r in results:
            score, _, meta_bytes, *_ = r
            meta = json.loads(meta_bytes)
            print(f"  -> {meta.get('text', '')} (score: {score:.4f})")
        print()
        print()
