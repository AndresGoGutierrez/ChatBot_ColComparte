import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Cargar artefactos
embeddings = np.load("data/embeddings/embeddings.npy")
with open("data/chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print(f"Corpus: {len(chunks)} chunks | Embeddings shape: {embeddings.shape}\n")

# Preguntas de prueba 
preguntas = [
    "¿Qué es Colombia Comparte?",
    "¿Cuáles son los programas de la organización?",
    "¿En qué países tiene presencia Colombia Comparte?",
    "¿Cuántas personas ha ayudado?",
    "¿Cuál es la misión de Colombia Comparte?",
]

for pregunta in preguntas:
    q_emb = model.encode([pregunta], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:2]

    print(f"{'='*60}")
    print(f"PREGUNTA: {pregunta}")
    for i, idx in enumerate(top_idx):
        print(f"  [{i+1}] score={scores[idx]:.3f} | {chunks[idx]['text'][:150]}...")
    print()