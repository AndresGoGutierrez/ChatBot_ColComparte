import sys
sys.path.insert(0, '.')
from src.chatbot import chat

PRUEBAS = [
    # (pregunta, categoria, keywords_esperados)
    ("¿Qué es Colombia Comparte?",
     "EN corpus",
     ["organización", "bienestar", "emprendimiento"]),

    ("¿Quiénes fundaron Colombia Comparte?",
     "EN corpus",
     ["Carolina", "Eduardo", "cofundadores"]),

    ("¿Qué es el programa EDIFICA?",
     "EN corpus",
     ["EDIFICA", "emprendimiento", "programa"]),

    ("¿Cuál es la misión de Colombia Comparte?",
     "EN corpus",
     ["misión", "bienestar", "transformación", "propósito"]),

    ("¿Cuánto cuesta el programa EDIFICA?",
     "FUERA corpus",
     ["no tengo", "información", "suficiente"]),

    ("¿Cuál es el NIT de Colombia Comparte?",
     "FUERA corpus",
     ["no tengo", "información", "suficiente"]),

    ("¿Quién es el presidente de Colombia?",
     "FUERA corpus",
     ["no tengo", "información", "suficiente"]),

    ("¿Cómo puedo contactar a Colombia Comparte?",
     "AMBIGUA",
     ["colombia comparte", "contacto", "no tengo"]),
]


def main():
    print("\nTEST SEMANA 3 — Generación con LLM\n")
    print("Primera ejecución puede tardar ~2 min (descarga del modelo)\n")

    correctas = 0

    for pregunta, categoria, keywords in PRUEBAS:
        print(f"{'='*60}")
        print(f"[{categoria}] {pregunta}")

        respuesta = chat(pregunta)
        respuesta_lower = respuesta.lower()

        acierto = any(kw.lower() in respuesta_lower for kw in keywords)
        correctas += int(acierto)
        estado = "B" if acierto else "X"

        print(f"{estado} RESPUESTA: {respuesta}")
        print(f"   Keywords esperados: {keywords}")
        print()

    accuracy = correctas / len(PRUEBAS) * 100
    print(f"{'='*60}")
    print(f"ACCURACY: {accuracy:.1f}%  ({correctas}/{len(PRUEBAS)})")

    if accuracy >= 80:
        print("Objetivo alcanzado — listo para Semana 4")
    elif accuracy >= 65:
        print("Aceptable — ajusta el prompt si alguna categoría falla")
    else:
        print("Revisar prompt o corpus antes de continuar")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()