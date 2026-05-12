# src/generator.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.logger import log

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK  = "No tengo suficiente información para responder esa pregunta con los datos disponibles."

_tokenizer = None
_model     = None
_device    = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Cargando modelo en: {_device.upper()}", "INFO")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    _model.eval()
    log(f"Modelo cargado en {_device.upper()}. Listo.", "SUCCESS")


def build_messages(query: str, context: str) -> list[dict]:
    system = (
        "Eres el chatbot oficial de Colombia Comparte. "
        "REGLA ABSOLUTA: Responde ÚNICAMENTE con información que aparezca "
        "de forma LITERAL en el contexto recuperado. "
        "PROHIBIDO inventar nombres, cifras, fechas, números, pasos o datos. "
        "PROHIBIDO usar conocimiento externo al contexto proporcionado. "
        "PROHIBIDO inventar formas de contacto, redes sociales o direcciones. "
        "PROHIBIDO generar listas de pasos si no están en el contexto. "
        "Si la información no está en el contexto, responde EXACTAMENTE: "
        f"'{FALLBACK}' — sin agregar nada más."
    )
    user = (
        "CONTEXTO RECUPERADO (única fuente de información permitida):\n"
        f"{context}\n\n"
        "VERIFICACIÓN OBLIGATORIA: Antes de responder, confirma que la información "
        "que vas a dar aparece literalmente en el contexto anterior. "
        "Si no aparece, di exactamente: "
        f"'{FALLBACK}'\n\n"
        f"PREGUNTA: {query}\n\n"
        "RESPUESTA (máximo 3 frases, solo con datos del contexto):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user}
    ]


def truncate_context_safely(results: list[dict], max_words: int = 800) -> str:
    """
    Construye el contexto con IDs y fuentes.
    Trunca por chunks completos, nunca a mitad de oración.
    """
    selected    = []
    total_words = 0

    for r in results:
        chunk_words = len(r['text'].split())
        if total_words + chunk_words > max_words:
            log(f"Contexto truncado antes de chunk {r['id']} "
                f"(acumulado: {total_words} palabras)", "WARN")
            break
        selected.append(f"[{r['id']} | fuente: {r['source']}]\n{r['text']}")
        total_words += chunk_words

    log(f"Palabras en contexto final: {total_words}", "INFO")
    return "\n\n---\n\n".join(selected)


def _verify_answer(answer: str, context: str) -> str:
    """
    Verificación post-generación.
    Detecta respuestas que contienen datos ausentes del contexto.
    Aplica fallback si la respuesta parece inventada.
    """
    answer_lower  = answer.lower()
    context_lower = context.lower()

    # Detectar números que parecen identificadores (NIT, teléfonos, etc.)
    # Si la respuesta tiene un número largo que no está en el contexto → fallback
    import re
    numeros_respuesta = re.findall(r'\b\d{3,}\b', answer)
    for numero in numeros_respuesta:
        if numero not in context:
            log(f"Número inventado detectado: {numero} → fallback", "WARN")
            return FALLBACK

    # Detectar si la respuesta tiene más de 3 palabras que no están en el contexto
    # Esto atrapa respuestas completamente desconectadas del contexto
    palabras_respuesta = set(re.findall(r'\b[a-záéíóúñ]{4,}\b', answer_lower))
    palabras_contexto  = set(re.findall(r'\b[a-záéíóúñ]{4,}\b', context_lower))
    palabras_nuevas    = palabras_respuesta - palabras_contexto

    # Si más del 40% de las palabras de la respuesta no están en el contexto
    # y la respuesta es corta (posiblemente un dato inventado) → fallback
    if len(palabras_respuesta) > 0:
        ratio_nuevo = len(palabras_nuevas) / len(palabras_respuesta)
        if ratio_nuevo > 0.60 and len(answer.split()) < 15:
            log(f"Respuesta con {ratio_nuevo:.0%} palabras fuera del contexto → fallback", "WARN")
            return FALLBACK

    return answer

def generate_answer(query: str, context: str) -> str:
    """
    Genera respuesta usando apply_chat_template.
    Forma correcta para modelos Instruct como Qwen2.5.
    """
    _load_model()

    log(f"Generando respuesta para: {query[:60]}...", "INFO")

    # Fallback inmediato si no hay contexto
    if not context or not context.strip():
        log("Contexto vacío → fallback activado", "WARN")
        return FALLBACK

    # Construir mensajes con formato chat
    messages = build_messages(query, context)

    # apply_chat_template formatea correctamente para Qwen2.5-Instruct
    inputs = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    log(f"Tokens del prompt: {inputs['input_ids'].shape[1]}", "INFO")

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.1,   # muy conservador — evita alucinaciones
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id
        )

    # Decodificar solo los tokens nuevos (no repetir el prompt)
    input_len  = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    raw_answer = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    log(f"Respuesta cruda: {raw_answer[:100]}", "INFO")

    answer = _clean_answer(raw_answer)
    answer = _verify_answer(answer, context)

    # Fallback si la respuesta quedó vacía o muy corta
    if len(answer.strip()) < 10:
        log("Respuesta demasiado corta → fallback", "WARN")
        return FALLBACK

    log(f"Respuesta generada ({len(answer)} chars)", "SUCCESS")
    return answer


def _clean_answer(text: str) -> str:
    """Elimina artefactos comunes en salidas de modelos Instruct."""
    text = text.strip()
    text = re.sub(r'^(Respuesta:|Asistente:|Assistant:)\s*',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() or FALLBACK