import sys
sys.path.insert(0, '.')
from src.retriever import retrieve_context
from src.logger import log

# ── Categorías de preguntas ──────────────────────────────────────────────────

PREGUNTAS_EN_CORPUS = [
    "¿Qué es Colombia Comparte?",
    "¿Cuál es la misión de Colombia Comparte?",
    "¿Qué es el programa EDIFICA?",
    "¿Cuáles son los programas que ofrece Colombia Comparte?",
    "¿Cómo nació Colombia Comparte?",
    "¿Quiénes fundaron Colombia Comparte?",
    "¿Qué es la pobreza oculta?",
    "¿En qué países tiene presencia?",
]

PREGUNTAS_FUERA_CORPUS = [
    "¿Cuál es el número de teléfono de Colombia Comparte?",
    "¿Cuánto cuesta inscribirse a EDIFICA?",
    "¿Cuál es el NIT de la fundación?",
    "¿Quién es el presidente de Colombia en 2024?",
]

PREGUNTAS_AMBIGUAS = [
    "¿Cómo puedo ayudar?",
    "¿Qué hace?",
    "quiero información",
    "cuéntame más",
]


def evaluar_categoria(nombre: str, preguntas: list[str],
                       espera_resultados: bool, umbral_minimo: float = 0.25):
    print(f"\n{'='*65}")
    print(f"CATEGORÍA: {nombre}")
    print(f"{'='*65}")

    correctas = 0
    for pregunta in preguntas:
        resultado = retrieve_context(pregunta)
        tiene_resultados = len(resultado['results']) > 0
        top_score = resultado['results'][0]['score'] if tiene_resultados else 0.0
        top_fuente = resultado['results'][0]['source'] if tiene_resultados else 'ninguna'

        # Para preguntas EN corpus
        # Para preguntas FUERA
        if espera_resultados:
            acierto = tiene_resultados and top_score >= umbral_minimo
        else:
            acierto = not tiene_resultados or top_score < 0.40

        estado = "B" if acierto else "x"
        correctas += int(acierto)

        print(f"\n  {estado} [{top_score:.3f}] {pregunta[:55]}...")
        if tiene_resultados:
            print(f"      fuente: {top_fuente}")
            print(f"      texto:  {resultado['results'][0]['text'][:100]}...")
        else:
            print(f"      → Sin resultados (activará fallback)")

    pct = correctas / len(preguntas) * 100
    print(f"\n  Resultado: {correctas}/{len(preguntas)} ({pct:.0f}%)")
    return correctas, len(preguntas)


def main():
    print("\n TEST SEMANA 2 — Motor de Recuperación\n")

    total_correctas = 0
    total_preguntas = 0

    c, t = evaluar_categoria(
        "Preguntas EN corpus", PREGUNTAS_EN_CORPUS,
        espera_resultados=True
    )
    total_correctas += c; total_preguntas += t

    c, t = evaluar_categoria(
        "Preguntas FUERA del corpus", PREGUNTAS_FUERA_CORPUS,
        espera_resultados=False
    )
    total_correctas += c; total_preguntas += t

    c, t = evaluar_categoria(
        "Preguntas AMBIGUAS", PREGUNTAS_AMBIGUAS,
        espera_resultados=False
    )
    total_correctas += c; total_preguntas += t

    accuracy = total_correctas / total_preguntas * 100
    print(f"\n{'='*65}")
    print(f"ACCURACY RETRIEVER: {accuracy:.1f}%  ({total_correctas}/{total_preguntas})")

    if accuracy >= 75:
        print("Retriever listo para Semana 3")
    elif accuracy >= 60:
        print("Retriever aceptable — ajusta threshold si es necesario")
    else:
        print("Retriever necesita ajustes antes de continuar")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()