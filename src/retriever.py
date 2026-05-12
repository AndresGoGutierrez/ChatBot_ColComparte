# src/retriever.py
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from src.logger import log

EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
CHUNKS_PATH     = "data/chunks/chunks.json"
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_K       = 4       # igual que el docente
SCORE_THRESHOLD = 0.25    # igual que el docente

_model      = None
_embeddings = None
_chunks     = None


def _load_resources():
    global _model, _embeddings, _chunks
    if _model is None:
        log("Cargando recursos del retriever...", "INFO")
        _model      = SentenceTransformer(MODEL_NAME)
        _embeddings = np.load(EMBEDDINGS_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            _chunks = json.load(f)
        log(f"Recursos cargados. {len(_chunks)} chunks disponibles.", "SUCCESS")


def retrieve_context(query: str, k: int = DEFAULT_K,
                     threshold: float = SCORE_THRESHOLD) -> dict:
    """
    Dado un query devuelve los k fragmentos más similares.

    Retorna:
        query:        pregunta original
        results:      lista de {id, text, source, score}
        context_text: texto listo para el prompt con IDs incluidos
        found:        bool — True si hay al menos un resultado útil
    """
    _load_resources()

    log(f"QUERY: {query}", "QUERY")

    # 1. Embed la query
    q_emb  = _model.encode([query], normalize_embeddings=True)

    # 2. Similitud coseno (producto punto porque embeddings están normalizados)
    scores = (_embeddings @ q_emb.T).flatten()

    # 3. Top-k ordenados por score descendente
    top_indices = np.argsort(scores)[::-1][:k]

    # 4. Filtrar por umbral
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= threshold:
            results.append({
                "id":     int(idx),
                "text":   _chunks[idx]['text'],
                "source": _chunks[idx]['source'],
                "score":  round(score, 4)
            })

    # 5. Logging detallado (útil en sustentación)
    log(f"Top scores: {[r['score'] for r in results]}", "SCORE")
    log(f"Chunks recuperados: {len(results)} de {k} solicitados", "INFO")

    if not results:
        log("Sin resultados sobre el umbral → fallback en generator", "WARN")

    # 6. Construir context_text con IDs (adoptado del docente)
    context_text = "\n\n---\n\n".join(
        [f"[{r['id']} | fuente: {r['source']}]\n{r['text']}" for r in results]
    )

    return {
        "query":        query,
        "results":      results,
        "context_text": context_text,
        "found":        len(results) > 0
    }


def format_context(results: list[dict], max_chars: int = 4000) -> str:
    """
    Versión del docente: trunca por caracteres con IDs visibles.
    Úsala como alternativa a truncate_context_safely() si el prompt
    supera el límite de tokens.
    """
    blocks    = []
    used_chars = 0

    for r in results:
        block = f"[{r['id']} | score={r['score']:.4f}]\n{r['text']}"
        if used_chars + len(block) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block)

    return "\n\n---\n\n".join(blocks)