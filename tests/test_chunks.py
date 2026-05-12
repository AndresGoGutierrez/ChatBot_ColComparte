import json

with open("data/chunks/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Chunks ordenados por tamaño (de menor a mayor):\n")

for c in sorted(chunks, key=lambda x: x["word_count"])[:5]:
    print(f"ID {c['id']} | {c['word_count']} palabras | fuente: {c['source']}")
    print(f"Texto: {c['text'][:200]}...\n")