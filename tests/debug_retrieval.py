import json
import numpy as np
from sentence_transformers import SentenceTransformer

embeddings = np.load("data/embeddings/embeddings.npy")

with open("data/chunks/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

query = "¿Cuántas familias han sido impactadas?"
q_emb = model.encode([query], normalize_embeddings=True)

scores = (embeddings @ q_emb.T).flatten()

print("TOP 5 chunks:")
for idx in np.argsort(scores)[::-1][:5]:
    print(f"score={scores[idx]:.3f} | fuente={chunks[idx]['source']} | palabras={chunks[idx]['word_count']}")
    print(chunks[idx]["text"][:200])
    print()