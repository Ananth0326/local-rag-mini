import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the same model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load vectors.json
with open('vectors.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chunks = data['chunks']
embeddings = np.array(data['embeddings'])

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
    
    norm_a = 0
    for i in range(len(a)):
        norm_a += a[i] * a[i]
    norm_a = norm_a ** 0.5
    
    norm_b = 0
    for i in range(len(b)):
        norm_b += b[i] * b[i]
    norm_b = norm_b ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

# Get user question
query = input("Ask: ")
query_embedding = model.encode([query])[0]

# Compare with all chunks
scores = []
for i, emb in enumerate(embeddings):
    sim = cosine_similarity(query_embedding, emb)
    scores.append((sim, i))

# Sort by highest score
scores.sort(reverse=True)

# Show results
print("\n" + "="*60)
print(f"Question: {query}")
print("="*60)

for sim, idx in scores[:3]:
    print(f"\n[Score: {sim:.4f}]")
    print(chunks[idx])
    print("-"*60)