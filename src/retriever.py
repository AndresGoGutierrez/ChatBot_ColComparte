# src/retriever.py
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from src.logger import log

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
CHUNKS_PATH     = "data/chunks/chunks.json"
MODEL_NAME      = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_K       = 2
SCORE_THRESHOLD = 0.30
BM25_ALPHA      = 0.72   # peso semántico; 0.28 queda para BM25

_model      = None
_embeddings = None
_chunks     = None
_bm25       = None


_STOPWORDS_ES = {
    "del", "los", "las", "son", "que", "con", "una", "uno", "por",
    "para", "como", "sus", "hay", "han", "has", "fue", "ser", "est",
    "esta", "este", "eso", "ese", "esa", "pero", "más", "mas", "muy",
    "sin", "sobre", "entre", "cuando", "donde", "cual", "cuales",
    "todo", "toda", "todos", "todas", "cada", "otro", "otra", "otros",
    "también", "tambien", "puede", "tienen", "tiene", "hacer", "hace",
}


def _tokenize_bm25(text: str) -> list:
    """Tokenizador BM25: minúsculas, sin puntuación, >3 chars, sin stopwords."""
    text = text.lower()
    text = re.sub(r'[^\wáéíóúüñ\s]', ' ', text)
    return [t for t in text.split() if len(t) > 3 and t not in _STOPWORDS_ES]


def _load_resources():
    global _model, _embeddings, _chunks, _bm25
    if _model is None:
        log("Cargando recursos del retriever...", "INFO")
        _model      = SentenceTransformer(MODEL_NAME)
        _embeddings = np.load(EMBEDDINGS_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            _chunks = json.load(f)

        if _HAS_BM25:
            tokenized = [_tokenize_bm25(c['text']) for c in _chunks]
            _bm25 = BM25Okapi(tokenized)
            log("Índice BM25 construido (búsqueda híbrida activa).", "INFO")
        else:
            log("rank_bm25 no disponible → solo búsqueda semántica.", "WARN")

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
    q_emb   = _model.encode([query], normalize_embeddings=True)

    # 2. Similitud coseno semántica
    sem_scores = (_embeddings @ q_emb.T).flatten()

    # 3. BM25 léxico (si está disponible)
    if _HAS_BM25 and _bm25 is not None:
        bm25_raw = np.array(_bm25.get_scores(_tokenize_bm25(query)))
        bm25_max = bm25_raw.max()
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw
        scores = BM25_ALPHA * sem_scores + (1 - BM25_ALPHA) * bm25_norm
        log(f"Búsqueda híbrida (α={BM25_ALPHA} semántica + {1-BM25_ALPHA:.2f} BM25)", "INFO")
    else:
        scores = sem_scores

    # 4. Top-k ordenados por score combinado descendente
    top_indices = np.argsort(scores)[::-1][:k]

    # 5. Filtrar por umbral (se usa el score semántico como referencia de calidad)
    results = []
    for idx in top_indices:
        score     = float(scores[idx])
        sem_score = float(sem_scores[idx])
        if sem_score >= threshold:   # umbral sobre score semántico puro
            results.append({
                "id":        int(idx),
                "text":      _chunks[idx]['text'],
                "source":    _chunks[idx]['source'],
                "score":     round(score, 4),
                "sem_score": round(sem_score, 4),
            })

    # 6. Logging detallado (útil en sustentación)
    log(f"Top scores (híbrido): {[r['score'] for r in results]}", "SCORE")
    log(f"Top scores (semántico): {[r['sem_score'] for r in results]}", "SCORE")
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