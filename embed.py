import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data/sample.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    def chunk_text(text, chunk_size=500, overlap=50):
     chunks = []
     step = chunk_size - overlap
    
     if len(text) <= chunk_size:
        return [text]
    
     for i in range(0, len(text) - chunk_size + 1, step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
     if len(text) % step != 0:
        last_chunk = text[-chunk_size:]
        chunks.append(last_chunk)
    
     return chunks
chunks = chunk_text(text)
print(f"Created {len(chunks)} chunks")

print("Creating embeddings...")
embeddings = model.encode(chunks)
print(f"Embeddings shape: {embeddings.shape}")

data = {
    "chunks": chunks,
    "embeddings": embeddings.tolist()
}

with open('vectors.json', 'w', encoding='utf-8') as f:
    json.dump(data, f)

print("Saved to vectors.json")