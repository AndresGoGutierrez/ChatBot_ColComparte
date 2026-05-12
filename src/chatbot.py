# src/chatbot.py
from src.retriever import retrieve_context
from src.generator import generate_answer, truncate_context_safely, FALLBACK
from src.logger import log


def chat(query: str, k: int = 4) -> str:
    """Pipeline completo RAG."""
    if not query or len(query.strip()) < 3:
        return "Por favor escribe una pregunta válida."

    log("=== NUEVA QUERY ===", "INFO")

    # Paso 1: recuperar contexto
    retrieval = retrieve_context(query, k=k)

    # Paso 2: truncar de forma segura
    context = truncate_context_safely(retrieval["results"], max_words=800)

    # Paso 3: generar respuesta
    answer = generate_answer(query, context)

    log("Pipeline completado.", "SUCCESS")
    return answer