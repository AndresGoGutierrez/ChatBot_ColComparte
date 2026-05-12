import json
import numpy as np
from sentence_transformers import SentenceTransformer

embeddings = np.load("data/embeddings/embeddings.npy")

with open("data/chunks/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

preguntas = [
    ("¿Cuántas familias han sido impactadas?", "numérica"),
    ("¿Cuántas personas ha ayudado Colombia Comparte?", "numérica variante"),
    ("impacto familias cifras Colombia Comparte", "keywords directos"),
    ("resultados logros Colombia Comparte", "semántica amplia"),
]

for pregunta, tipo in preguntas:
    q = model.encode([pregunta], normalize_embeddings=True)
    scores = (embeddings @ q.T).flatten()
    top = np.argsort(scores)[::-1][0]

    print(f"[{tipo}]")
    print(f"  Query: {pregunta}")
    print(f"  Top score: {scores[top]:.3f} | fuente: {chunks[top]['source']}")
    print(f"  Texto: {chunks[top]['text'][:120]}...")
    print()