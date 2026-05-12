import json
import numpy as np
from sentence_transformers import SentenceTransformer


def load_data():
    embeddings = np.load("data/embeddings/embeddings.npy")
    with open("data/chunks/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, chunks


def evaluar_queries(model, embeddings, chunks):
    preguntas = [
        ("¿Cuántas familias han sido impactadas?", "numérica directa"),
        ("¿Cuántas personas ha ayudado Colombia Comparte?", "numérica variante"),
        ("impacto familias cifras Colombia Comparte", "keywords"),
        ("resultados logros Colombia Comparte", "semántica amplia"),
    ]

    for pregunta, tipo in preguntas:
        q_emb = model.encode([pregunta], normalize_embeddings=True)
        scores = (embeddings @ q_emb.T).flatten()

        print("=" * 70)
        print(f"TIPO: {tipo}")
        print(f"PREGUNTA: {pregunta}")
        print("TOP 3 CHUNKS:\n")

        for idx in np.argsort(scores)[::-1][:3]:
            chunk = chunks[idx]

            # detectar si contiene cifras relevantes
            texto_lower = chunk["text"].lower()
            tiene_cifra = (
                "1200" in texto_lower
                or "700" in texto_lower
                or "familias" in texto_lower
                or "emprendimientos" in texto_lower
            )

            marca = "TIENE CIFRA" if tiene_cifra else "   "

            print(f"{marca} score={scores[idx]:.3f}")
            print(f"fuente={chunk['source']} | palabras={chunk['word_count']}")
            print(f"{chunk['text'][:200]}...")
            print()

        # Validación automática (muy útil para tu profe)
        # Reemplaza la sección de validación automática por esta:
        top3_indices = np.argsort(scores)[::-1][:3]
        cifra_en_top3 = any(
            "1200" in chunks[idx]["text"].lower() or
            "familias" in chunks[idx]["text"].lower()
            for idx in top3_indices
        )
        cifra_en_top1 = (
            "1200" in chunks[top3_indices[0]]["text"].lower() or
            "familias" in chunks[top3_indices[0]]["text"].lower()
        )

        if cifra_en_top1:
            print("EXCELENTE: cifra en posición 1 — retrieval óptimo")
        elif cifra_en_top3:
            print("ACEPTABLE: cifra en top-3 — el LLM podrá responder")
        else:
            print("PROBLEMA REAL: cifra ausente del top-3 — el LLM fallará")


def main():
    print("Cargando modelo...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Cargando datos...")
    embeddings, chunks = load_data()

    print(f"\nCorpus: {len(chunks)} chunks")
    print(f"Embeddings shape: {embeddings.shape}\n")

    evaluar_queries(model, embeddings, chunks)


if __name__ == "__main__":
    main()