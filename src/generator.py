# src/generator.py
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from src.logger import log

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
NLI_ID   = "cross-encoder/nli-deberta-v3-small"
FALLBACK  = "No tengo suficiente información para responder esa pregunta con los datos disponibles."

# Modelo generador
_tokenizer = None
_model     = None
_device    = None

# Modelo verificador NLI
_nli_tokenizer = None
_nli_model     = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Cargando modelo en: {_device.upper()}", "INFO")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # bfloat16 en CPU reduce RAM a ~6 GB (float32 necesitaría ~12 GB y agota RAM)
    dtype = torch.float16 if _device == "cuda" else torch.bfloat16
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    _model.eval()
    log(f"Modelo cargado en {_device.upper()}. Listo.", "SUCCESS")


def _load_verifier():
    global _nli_tokenizer, _nli_model
    if _nli_model is not None:
        return
    log("Cargando verificador NLI (cross-encoder/nli-deberta-v3-small)...", "INFO")
    _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_ID)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_ID)
    _nli_model.eval()
    log("Verificador NLI cargado.", "SUCCESS")


def verify_with_classifier(context: str, answer: str) -> bool:
    _load_verifier()
    ctx_trunc = " ".join(context.split()[:300])
    features = _nli_tokenizer(
        ctx_trunc, answer,
        padding=True, truncation=True,
        return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        logits = _nli_model(**features).logits
    probs    = logits.softmax(dim=1)[0]
    id2label = _nli_model.config.id2label
    label_map = {id2label[i].lower(): probs[i].item() for i in range(len(probs))}
    entail = label_map.get("entailment", 0.0)
    contra = label_map.get("contradiction", 0.0)
    log(f"NLI: entailment={entail:.2f}  contradiction={contra:.2f}", "INFO")
    return entail > 0.40


def build_messages(query: str, context: str) -> list[dict]:
    system = (
        "Eres el asistente oficial de Latinoamérica Comparte y Colombia Comparte. "
        "SOLO puedes responder usando datos LITERALES del contexto. "
        "PROHIBIDO inventar, resumir, parafrasear, completar, especular o agregar información. "
        "NO repitas la pregunta, NO hagas listas, NO uses frases genéricas. "
        "Si la respuesta no está en el contexto, responde exactamente: '" + FALLBACK + "'"
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


def truncate_context_safely(results: list[dict], max_words: int = 220) -> str:
    selected    = []
    total_words = 0
    for i, r in enumerate(results):
        chunk_words = len(r['text'].split())
        if total_words + chunk_words > max_words:
            if i == 0:
                words = r['text'].split()[:max_words]
                text  = ' '.join(words)
                selected.append(f"[{r['id']} | fuente: {r['source']}]\n{text}")
                total_words = max_words
            else:
                log(f"Contexto truncado antes de chunk {r['id']} (acumulado: {total_words} palabras)", "WARN")
            break
        selected.append(f"[{r['id']} | fuente: {r['source']}]\n{r['text']}")
        total_words += chunk_words
    log(f"Palabras en contexto final: {total_words}", "INFO")
    return "\n\n---\n\n".join(selected)


def _verify_answer(answer: str, context: str) -> str:
    answer_lower  = answer.lower()
    context_lower = context.lower()
    numeros = re.findall(r'\b\d{3,}\b', answer)
    for num in numeros:
        if num not in context:
            log(f"Número inventado detectado: {num} → fallback", "WARN")
            return FALLBACK
    palabras_resp = set(re.findall(r'\b[a-záéíóúñ]{4,}\b', answer_lower))
    palabras_ctx  = set(re.findall(r'\b[a-záéíóúñ]{4,}\b', context_lower))
    palabras_new  = palabras_resp - palabras_ctx
    if len(palabras_resp) > 0:
        ratio = len(palabras_new) / len(palabras_resp)
        if ratio > 0.40:
            log(f"Verificación léxica: {ratio:.0%} palabras fuera de contexto → fallback", "WARN")
            return FALLBACK
    return answer


def _clean_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^(Respuesta:|Asistente:|Assistant:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip() or FALLBACK


def generate_answer(query: str, context: str) -> str:
    _load_model()
    log(f"Generando respuesta para: {query[:60]}...", "INFO")
    if not context or not context.strip():
        log("Contexto vacío → fallback activado", "WARN")
        return FALLBACK
    messages = build_messages(query, context)
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
            max_new_tokens=130,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=_tokenizer.eos_token_id
        )
    input_len  = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    raw_answer = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    log(f"Respuesta cruda: {raw_answer[:100]}", "INFO")
    answer = _clean_answer(raw_answer)
    answer = _verify_answer(answer, context)
    if answer != FALLBACK and len(answer.strip()) > 10:
        try:
            if not verify_with_classifier(context, answer):
                log("Verificador NLI: respuesta NO sustentada → fallback", "WARN")
                return FALLBACK
        except Exception as e:
            log(f"Error en verificador NLI (ignorado): {e}", "ERROR")
    if len(answer.strip()) < 10:
        log("Respuesta demasiado corta → fallback", "WARN")
        return FALLBACK
    log(f"Respuesta generada ({len(answer)} chars)", "SUCCESS")
    return answer
