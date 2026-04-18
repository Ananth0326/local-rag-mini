# 🧠 local-rag-mini

A minimal Retrieval-Augmented Generation (RAG) system built from scratch using Python.

No LangChain. No vector databases. Just embeddings, cosine similarity, and core retrieval logic.

---

## 🔍 What This Project Does

This project implements the core retrieval layer of a RAG system:

- Converts documents into embeddings
- Stores them locally
- Retrieves relevant context using cosine similarity

This mirrors how production RAG systems fetch context before passing it to an LLM.

---

## ⚙️ How It Works

### 1. Indexing (embed.py)
- Load raw text
- Chunk into 500 characters with 50 overlap
- Generate embeddings using all-MiniLM-L6-v2
- Store vectors in JSON

### 2. Querying (search.py)
- Convert query into embedding
- Compute cosine similarity against stored vectors
- Return top-k relevant chunks

---

## 🛠️ Tech Stack

- Python
- sentence-transformers
- NumPy

Embedding Model: all-MiniLM-L6-v2 (384-dim)  
Similarity: Cosine similarity (implemented manually)  
Storage: JSON (no external DB)

---

## 📊 Example

Query: what is RAG?

Top Result:
RAG stands for Retrieval-Augmented Generation. It combines retrieval with language models to generate context-aware responses.

---

## 🧠 Key Learnings

- Embeddings represent semantic meaning as vectors
- Cosine similarity measures similarity between vectors
- Chunking with overlap preserves context
- Retrieval is the foundation of RAG systems

---

## ⚠️ Limitations

- No vector database (not scalable for large datasets)
- Retrieval only (no LLM generation yet)
- No ranking refinement or filtering

---

## 🔜 Next Steps

- Add LLM (Groq) for answer generation
- Build FastAPI backend for querying
- Add caching and ranking improvements

---

## 📁 Project Structure

local-rag-mini/
├── embed.py
├── search.py
├── data/
└── vectors.json

---

## 🚀 Run It

# Setup
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create embeddings
python embed.py

# Ask questions
python search.py