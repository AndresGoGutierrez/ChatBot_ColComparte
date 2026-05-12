import sys

sys.path.insert(0, '.')

from src.preprocess import drop_structural_headers

def main():
    with open('data/raw/web_latam.txt', encoding='utf-8') as f:
        original = f.read()

    limpio = drop_structural_headers(original)

    print('=== PRIMERAS 500 CHARS DESPUÉS DE LIMPIAR ===')
    print(limpio[:500])


if __name__ == "__main__":
    main()