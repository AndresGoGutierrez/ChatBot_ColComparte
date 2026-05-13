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
    # float16 solo si hay GPU — en CPU usar float32 para estabilidad
    dtype = torch.float16 if _device == "cuda" else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    _model.eval()
    log(f"Modelo cargado en {_device.upper()}. Listo.", "SUCCESS")


def build_messages(query: str, context: str) -> list[dict]:
    system = (
        "Eres el asistente oficial de Latinoamérica Comparte y Colombia Comparte. "
        "Responde DIRECTAMENTE con los datos exactos del contexto. "
        "REGLAS ESTRICTAS: 1) USA solo datos que aparezcan textualmente en el contexto. "
        "2) NO hagas listas de redes sociales ni plataformas digitales a menos que se pregunte explícitamente por redes sociales. "
        "3) Si preguntan por contacto o teléfono, responde SOLO con el número de teléfono y correo del contexto. "
        "4) NO uses frases como 'la pregunta indica', 'esto sugiere' o 'la plataforma'. "
        "5) Si la respuesta no está en el contexto, di exactamente: "
        f"'{FALLBACK}'"
    )
    user = (
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {query}\n"
        "RESPUESTA (máximo 2 frases, solo con datos literales del contexto):"
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

    for i, r in enumerate(results):
        chunk_words = len(r['text'].split())
        if total_words + chunk_words > max_words:
            if i == 0:
                # Incluir siempre al menos el primer chunk, recortando el texto
                words = r['text'].split()[:max_words]
                text  = ' '.join(words)
                selected.append(f"[{r['id']} | fuente: {r['source']}]\n{text}")
                total_words = max_words
            else:
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
        if ratio_nuevo > 0.75 and len(answer.split()) < 12:
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
            max_new_tokens=900,
            do_sample=False,          # greedy decoding: ~2x más rápido en CPU
            repetition_penalty=1.15,
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