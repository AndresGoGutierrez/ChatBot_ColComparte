import json

with open("data/chunks/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print("Chunks de web_principal (en orden):\n")

for c in chunks:
    if c["source"] == "web_principal":
        print(f"--- ID={c['id']} | {c['word_count']} palabras ---")
        print(c["text"][:300])
        print("...")
        print(f"[FINAL]: ...{c['text'][-150:]}")
        print()