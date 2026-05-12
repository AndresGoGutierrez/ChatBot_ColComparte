import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from src.logger import log

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_embeddings(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Convierte cada chunk en un vector de embedding normalizado."""
    
    log(f"Cargando modelo: {model_name}", "INFO")
    model = SentenceTransformer(model_name)

    texts = [chunk['text'] for chunk in chunks]

    log(f"Generando embeddings para {len(texts)} chunks...", "INFO")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    log(f"Embeddings generados. Shape: {embeddings.shape}", "SUCCESS")
    return embeddings


def attach_embeddings_to_chunks(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """
    Opcional (pero muy útil): agrega el embedding dentro de cada chunk
    para debugging o inspección.
    """
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    return chunks


def save_artifacts(
    embeddings: np.ndarray,
    chunks: list[dict],
    emb_path: str,
    chunks_path_out: str
):
    """Guarda embeddings y chunks sincronizados."""
    
    os.makedirs(os.path.dirname(emb_path), exist_ok=True)

    # Guardar embeddings como numpy
    np.save(emb_path, embeddings)

    # Guardar chunks (sin embeddings para no inflar el archivo)
    with open(chunks_path_out, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    log(f"Embeddings guardados: {emb_path}", "SUCCESS")
    log(f"Chunks sincronizados guardados: {chunks_path_out}", "SUCCESS")


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    """
    EXTRA: Construye índice FAISS.
    """
    try:
        # type: ignore
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Producto interno (coseno porque ya normalizaste)

        index.add(embeddings.astype(np.float32))

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)

        log(f"Índice FAISS guardado: {index_path} ({index.ntotal} vectores)", "SUCCESS")

    except ImportError:
        log("FAISS no instalado. Usa: pip install faiss-cpu --break-system-packages", "WARN")


if __name__ == "__main__":
    chunks = load_chunks("data/chunks/chunks.json")

    embeddings = generate_embeddings(chunks)

    # Validación clave
    assert len(chunks) == len(embeddings), "Mismatch entre chunks y embeddings"

    save_artifacts(
        embeddings,
        chunks,
        emb_path="data/embeddings/embeddings.npy",
        chunks_path_out="data/chunks/chunks.json"
    )

    log(f"Proceso completado correctamente", "SUCCESS")