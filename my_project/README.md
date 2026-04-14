# AI Semantic Search using Endee Vector Database

A semantic search application that uses [Endee](https://github.com/endee-io/endee) as the vector database to store and retrieve AI-generated embeddings for natural language queries.

## Project Overview

This project demonstrates how to build a semantic search engine using Endee as the vector database backend. Unlike keyword search, this system understands the **meaning** of your query and returns the most relevant result even when exact words don't match.

For example, querying `"neural networks"` returns `"Deep learning is a part of machine learning using neural networks."` — because Endee finds the closest vector by meaning, not by keyword.

## Features

- Semantic search powered by `all-MiniLM-L6-v2` embeddings (384 dimensions)
- Endee vector database for fast cosine similarity search
- Interactive query loop — type any natural language question
- Stores document metadata alongside vectors in Endee

## System Design

```
data.txt (documents)
      |
SentenceTransformer (all-MiniLM-L6-v2)
      |
384-dimensional embeddings
      |
Endee Vector DB (localhost:8080)  ← cosine similarity index
      |
Query → embedding → Endee search → Best Match
```

## How It Uses Vector Search

1. Each document in `data.txt` is converted to a 384-dim vector using a transformer model
2. Vectors are stored in Endee via `/api/v1/index/create` and `/api/v1/index/{name}/vector/insert`
3. At query time, the query is embedded and sent to Endee's `/api/v1/index/{name}/search`
4. Endee returns the top-k most similar vectors using cosine similarity (HNSW index)

## How to Run

### Step 1: Start Endee (Docker required)

```bash
docker run --ulimit nofile=100000:100000 -p 8080:8080 \
  -v ./endee-data:/data --name endee-server \
  endeeio/endee-server:latest
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the app

```bash
python app.py
```

## Sample Output

```
Index: Index created successfully
Indexed: 

Semantic Search ready. Type a query (or 'quit' to exit):

Query: what is python
Best Match:
  -> Python is widely used for AI and machine learning applications. (score: 0.7216)

Query: neural networks
Best Match:
  -> Deep learning is a part of machine learning using neural networks. (score: 0.7891)

Query: robots thinking
Best Match:
  -> Artificial intelligence is the simulation of human intelligence in machines. (score: 0.6134)

Query: extracting knowledge from data
Best Match:
  -> Data science involves extracting insights from structured and unstructured data. (score: 0.8102)
```

## Tech Stack

| Component | Technology |
|---|---|
| Vector Database | Endee |
| Embedding Model | all-MiniLM-L6-v2 |
| Language | Python 3.12 |
| Deployment | Docker |

## Dependencies

```
sentence-transformers
requests
msgpack
```
