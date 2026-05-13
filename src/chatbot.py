# src/chatbot.py
from src.retriever import retrieve_context
from src.generator import generate_answer, truncate_context_safely, FALLBACK
from src.logger import log

# Expansión de términos del dominio: si la query contiene alguna clave,
# se agregan los sinónimos al final antes de embeder.
_EXPANSIONS: dict[str, list[str]] = {
    "qué es":          ["fundación", "organización", "descripción", "misión", "propósito"],
    "que es":          ["fundación", "organización", "descripción", "misión", "propósito"],
    "cuántas":         ["número", "cantidad", "cifras", "total"],
    "cuantas":         ["número", "cantidad", "cifras", "total"],
    "ayudado":         ["beneficiado", "impactado", "transformado", "familias"],
    "personas":        ["familias", "beneficiarios", "impactados"],
    "países":          ["presencia", "latinoamérica", "ecuador", "chile", "argentina"],
    "paises":          ["presencia", "latinoamérica", "ecuador", "chile", "argentina"],
    "precio":          ["costo", "inversión", "valor", "inscripción"],
    "costo":           ["precio", "inversión", "valor"],
    "inscribir":       ["inscripción", "formulario", "participar", "registro"],
    "inscripción":     ["formulario", "participar", "registro"],
    "fundadores":      ["cofundadores", "carolina", "eduardo", "historia"],
    "fundador":        ["cofundadores", "carolina", "eduardo", "historia"],
    "fundó":           ["cofundadores", "carolina", "eduardo", "creadores"],
    "fundo":           ["cofundadores", "carolina", "eduardo", "creadores"],
    "creó":            ["fundadores", "carolina", "eduardo", "origen"],
    "creo":            ["fundadores", "carolina", "eduardo", "origen"],
    "contactar":       ["teléfono", "correo", "número", "llamar", "contacto"],
    "contacto":        ["teléfono", "correo", "número", "llamar"],
    "latinoamerica":   ["red", "comparte", "organización", "misión", "continental"],
    "latinoamérica":   ["red", "comparte", "organización", "misión", "continental"],
    "donación":        ["donar", "bancolombia", "cuenta", "aporte"],
    "donar":           ["donación", "bancolombia", "cuenta", "aporte"],
    "misión":          ["propósito", "objetivo", "transformar", "visión"],
    # Nueva arquitectura de marca
    "programa":        ["comparte academia", "deskubre", "estructura", "liderazgo", "talento"],
    "programas":       ["comparte academia", "deskubre", "estructura", "liderazgo", "talento"],
    "edifica":         ["comparte academia", "deskubre", "estructura", "emprendimiento"],
    "academia":        ["deskubre", "estructura", "emprendimiento", "comparte academia"],
    "deskubre":        ["comparte academia", "inicial", "exploración", "emprendimiento", "1 mes"],
    "descubre":        ["deskubre", "comparte academia", "inicial", "exploración"],
    "estructura":      ["comparte academia", "avanzado", "12 meses", "modelo de negocio"],
    "liderazgo":       ["comparte liderazgo", "cultura organizacional", "bienestar", "empresas"],
    "talento":         ["comparte talento", "conferencistas", "speakers", "eventos", "top speakers"],
    "top speakers":    ["comparte talento", "conferencistas", "speakers", "conferencias"],
    "conferencias":    ["comparte talento", "speakers", "eventos", "conferencistas"],
    "impacto":         ["familias", "transformadas", "cifras", "resultados"],
    "equipo":          ["mentores", "coaches", "conferencistas", "team"],
    "conferencista":   ["speaker", "conferencia", "comparte talento"],
    "shows":           ["entretenimiento", "comedy", "musical", "eventos", "comparte talento"],
    "eventos":         ["conferencias", "comparte talento", "cohorte", "actividades", "red que transforma"],
    "evento":          ["conferencias", "comparte talento", "cohorte", "actividades"],
    "recientes":       ["noticias", "novedades", "últimas", "eventos", "actividades"],
    "actividades":     ["eventos", "conferencias", "cohortes", "comparte talento"],
    "próximos":        ["eventos", "cohorte", "inscripción", "comparte academia"],
}


def _expand_query(query: str) -> str:
    """
    Añade términos relacionados al dominio si detecta palabras clave.
    No modifica el texto original; solo agrega contexto al final.
    """
    query_lower = query.lower()
    extras: list[str] = []
    for keyword, synonyms in _EXPANSIONS.items():
        if keyword in query_lower:
            extras.extend(synonyms)

    if extras:
        unique_extras = list(dict.fromkeys(extras))   # deduplicar manteniendo orden
        expanded = query + " " + " ".join(unique_extras)
        log(f"Query expandida: {expanded[:120]}", "INFO")
        return expanded
    return query


def chat(query: str, k: int = 3) -> str:
    """Pipeline completo RAG con query expansion."""
    if not query or len(query.strip()) < 3:
        return "Por favor escribe una pregunta válida."

    log("=== NUEVA QUERY ===", "INFO")

    # Paso 1: expandir query antes del embedding
    expanded = _expand_query(query)

    # Paso 2: recuperar contexto (con query expandida para retrieval)
    retrieval = retrieve_context(expanded, k=k)

    # Paso 3: truncar de forma segura (220 palabras ≈ 350 tokens → ~30s en CPU)
    context = truncate_context_safely(retrieval["results"], max_words=220)

    # Paso 4: generar respuesta (con query ORIGINAL para el LLM)
    answer = generate_answer(query, context)

    log("Pipeline completado.", "SUCCESS")
    return answer


def chat_with_sources(query: str, k: int = 3) -> tuple[str, list[dict]]:
    """
    Igual que chat() pero devuelve también los chunks recuperados.
    Usado por la interfaz Gradio para mostrar las fuentes.
    """
    if not query or len(query.strip()) < 3:
        return "Por favor escribe una pregunta válida.", []

    log("=== NUEVA QUERY (con fuentes) ===", "INFO")

    expanded  = _expand_query(query)
    retrieval = retrieve_context(expanded, k=k)
    context   = truncate_context_safely(retrieval["results"], max_words=220)
    answer    = generate_answer(query, context)

    log("Pipeline completado.", "SUCCESS")
    return answer, retrieval["results"]
