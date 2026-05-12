import sys
import os
import json

# Asegura acceso a la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    with open('data/chunks/chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)

    print('Buscando fundadores en todos los chunks:\n')

    keywords = ["carolina", "eduardo", "fundador"]

    encontrados = []

    for c in chunks:
        text = c.get("text", "").lower()

        if any(k in text for k in keywords):
            encontrados.append(c)

            print(f"ID={c.get('id')} | {c.get('word_count')} palabras | {c.get('source')}")
            print(c.get("text", "")[:300])
            print()

    print(f"Total encontrados: {len(encontrados)}")


if __name__ == "__main__":
    main()