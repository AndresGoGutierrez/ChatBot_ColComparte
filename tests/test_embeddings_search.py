import sys
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Asegura acceso a la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    embeddings = np.load('data/embeddings/embeddings.npy')

    with open('data/chunks/chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    query = "¿Quiénes fundaron Colombia Comparte?"
    q = model.encode([query], normalize_embeddings=True)

    scores = (embeddings @ q.T).flatten()

    print("Top 6 para fundadores:\n")

    for idx in np.argsort(scores)[::-1][:6]:
        text = chunks[idx].get("text", "").lower()

        tiene = (
            "carolina" in text or
            "eduardo" in text or
            "fundador" in text
        )

        marca = "B" if tiene else "  "

        print(f"{marca} ID={idx} score={scores[idx]:.3f} | {chunks[idx].get('source')}")


if __name__ == "__main__":
    main()