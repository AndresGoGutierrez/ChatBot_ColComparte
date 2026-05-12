import sys
import os
import json

# Para que funcione el import desde raíz del proyecto (buena práctica)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    with open('data/chunks/chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)

    print('Chunks de web_quienes_somos:\n')

    filtered = [x for x in chunks if x.get('source') == 'web_quienes_somos']

    for c in filtered:
        print(f"ID={c.get('id')} | {c.get('word_count')} palabras")
        print(c.get("text", "")[:300])
        print()


if __name__ == "__main__":
    main()