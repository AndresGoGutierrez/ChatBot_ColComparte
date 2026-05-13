# src/preprocess.py
import re
import json
import os
from src.logger import log


def clean_text(text: str) -> str:
    """Limpia el texto crudo manteniendo estructura útil para RAG."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s\.,;:¿?¡!\-áéíóúüñÁÉÍÓÚÜÑ@+()/]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str, max_words: int = 350, overlap: int = 50, source: str = "unknown"
) -> list[dict]:
    """
    Divide texto en chunks respetando oraciones.
    Ahora recibe el nombre de la fuente para rastreabilidad.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks = []
    current_words = []

    for sentence in sentences:
        words = sentence.split()
        if len(current_words) + len(words) > max_words and current_words:
            chunks.append(
                {
                    "text": " ".join(current_words),
                    "word_count": len(current_words),
                    "source": source,  # ← nombre del archivo real
                }
            )
            current_words = current_words[-overlap:]
        current_words.extend(words)

    if current_words:
        chunks.append(
            {
                "text": " ".join(current_words),
                "word_count": len(current_words),
                "source": source,
            }
        )

    return chunks

# src/preprocess.py — reemplaza SOLO la función load_all_sources

def drop_structural_headers(text: str) -> str:
    """
    Elimina encabezados estructurales de los archivos .txt.
    Maneja dos casos:
      - Etiqueta sola en su línea:  DESCRIPCIÓN GENERAL
      - Etiqueta con contenido:     TIPO: información regional
    """
    etiquetas = (
        r'DESCRIPCIÓN GENERAL|FUENTE|TIPO|ORGANIZACIÓN|ORIGEN'
        r'|ENFOQUE DE INTERVENCIÓN|ENFOQUE ACTUAL|ENFOQUE'
        r'|PROPÓSITO|ACLARACIÓN IMPORTANTE|MODELO DE IMPACTO'
        r'|IMPACTO ALCANZADO DE LA RED LATINOAMÉRICA COMPARTE'
        r'|IMPACTO ALCANZADO|IMPACTO Y CIFRAS DE TRANSFORMACIÓN|IMPACTO'
        r'|RED DE EMPRESAS Y ALIADOS|ACTIVIDADES Y CONTENIDOS'
        r'|PARTICIPACIÓN Y APOYO|RELACIÓN CON COLOMBIA COMPARTE'
        r'|PAÍSES DE LA RED|MISIÓN EN ACCIÓN|HISTORIA'
        r'|POBLACIÓN A LA QUE APOYA|TESTIMONIOS Y COMUNIDAD'
        r'|NOTICIAS Y ACTUALIDAD|SOBRE NOSOTROS'
        r'|PROGRAMA EDIFICA - ALTOS ESTUDIOS EN EMPRENDIMIENTO'
        r'|SHOWS Y ENTRETENIMIENTO EMPRESARIAL'
        r'|NODUS'
    )
    # Caso 1: línea que ES SOLO la etiqueta (con o sin dos puntos al final)
    text = re.sub(
        rf'^\s*({etiquetas})\s*:?\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    # Caso 2: línea que EMPIEZA con la etiqueta seguida de dos puntos y contenido
    text = re.sub(
        rf'^\s*({etiquetas})\s*:.*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    # Elimina separadores ---
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    # Elimina líneas entre corchetes [Fuente: ...]
    text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE)
    # Colapsar saltos múltiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_all_sources(raw_dir: str) -> list[dict]:
    """
    Carga cada archivo por separado, elimina ruido estructural
    y chunkea individualmente con su fuente correcta.
    """
    all_chunks = []
    chunk_id   = 0

    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.txt')])
    log(f"Archivos encontrados en {raw_dir}: {files}", "INFO")

    for filename in files:
        path = os.path.join(raw_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Paso 1: eliminar encabezados estructurales (NUEVO)
        content = drop_structural_headers(content)

        # Paso 2: limpieza general
        clean  = clean_text(content)

        # Paso 3: chunkear con fuente correcta
        source = filename.replace('.txt', '')
        chunks = split_into_chunks(clean, max_words=350, overlap=50, source=source)

        for chunk in chunks:
            chunk["id"] = chunk_id
            chunk_id   += 1

        all_chunks.extend(chunks)
        log(f"Cargado: {filename} → {len(chunks)} chunks", "SUCCESS")

    return all_chunks

def process_corpus(raw_dir: str, output_path: str):
    """Pipeline completo: carga por archivo → limpia → chunkea → guarda."""
    chunks = load_all_sources(raw_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    sizes = [c["word_count"] for c in chunks]
    log(f"{len(chunks)} chunks generados → {output_path}", "SUCCESS")
    log(f"Tamaño promedio: {sum(sizes)//len(sizes)} palabras", "INFO")
    log(f"Mínimo: {min(sizes)} · Máximo: {max(sizes)}", "INFO")

    # Resumen por fuente — muy útil para detectar archivos pequeños
    fuentes = {}
    for c in chunks:
        fuentes[c["source"]] = fuentes.get(c["source"], 0) + 1
    log("Chunks por fuente:", "INFO")
    for fuente, cantidad in fuentes.items():
        estado = "✅" if cantidad >= 2 else "⚠️  (archivo muy pequeño)"
        print(f"    {estado} {fuente}: {cantidad} chunk(s)")


if __name__ == "__main__":
    process_corpus("data/raw/", "data/chunks/chunks.json")
