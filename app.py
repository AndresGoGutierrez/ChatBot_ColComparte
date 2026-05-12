"""
app.py — Interfaz Gradio para el ChatBot de Colombia Comparte.
Ejecutar: python app.py
"""
import threading
import gradio as gr
from src.chatbot import chat_with_sources

# ---------------------------------------------------------------------------
# Pre-carga de modelos en hilo de fondo (evita espera en la primera consulta)
# ---------------------------------------------------------------------------
def _preload():
    from src.generator import _load_model
    from src.retriever import _load_resources
    _load_model()
    _load_resources()

threading.Thread(target=_preload, daemon=True).start()

# ---------------------------------------------------------------------------
# Preguntas de ejemplo
# ---------------------------------------------------------------------------
EXAMPLES = [
    "¿Qué es Colombia Comparte?",
    "¿Cuáles son los programas que ofrece la organización?",
    "¿En qué países tiene presencia Colombia Comparte?",
    "¿Cuántas personas ha ayudado Colombia Comparte?",
    "¿Cuál es la misión de Colombia Comparte?",
    "¿Cómo puedo inscribirme al programa EDIFICA?",
    "¿Quiénes son los fundadores de Colombia Comparte?",
    "¿Cómo puedo contactar a Colombia Comparte?",
    "¿Qué es la pobreza oculta?",
    "¿Qué conferencistas tiene Colombia Comparte?",
]

# ---------------------------------------------------------------------------
# Lógica del chat
# ---------------------------------------------------------------------------
def respond(message: str, history: list):
    """Llama al pipeline RAG y devuelve respuesta + fuentes formateadas."""
    if not message or not message.strip():
        return history, "", _format_sources([])

    answer, sources = chat_with_sources(message.strip())

    history = history + [[message, answer]]
    sources_md = _format_sources(sources)
    return history, "", sources_md


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "_Aún no se ha realizado ninguna consulta._"

    lines = ["### 📄 Fragmentos consultados\n"]
    for i, r in enumerate(sources, 1):
        score_bar = "🟢" if r["sem_score"] >= 0.60 else ("🟡" if r["sem_score"] >= 0.40 else "🔴")
        lines.append(
            f"**{i}. `{r['source']}`** — {score_bar} score `{r['sem_score']:.3f}`\n\n"
            f"> {r['text'][:280].strip()}{'...' if len(r['text']) > 280 else ''}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
CSS = """
#chatbot { height: 480px; }
#sources  { height: 480px; overflow-y: auto; }
footer { display: none !important; }
"""

with gr.Blocks(title="ChatBot Colombia Comparte") as demo:

    # ── Encabezado ──────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 🇨🇴 ChatBot Colombia Comparte
        Pregunta sobre la organización, sus programas, historia e impacto.
        *Powered by RAG · Qwen2.5-0.5B-Instruct · Búsqueda híbrida BM25 + semántica*
        """
    )

    # ── Cuerpo principal ────────────────────────────────────────────────────
    with gr.Row():
        # Panel izquierdo: chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversación",
                height=480,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Escribe tu pregunta aquí…",
                    label="",
                    scale=5,
                    lines=1,
                    autofocus=True,
                )
                send_btn = gr.Button("Enviar ▶", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑 Limpiar conversación", size="sm")

        # Panel derecho: fuentes
        with gr.Column(scale=2):
            sources_box = gr.Markdown(
                value="_Las fuentes utilizadas aparecerán aquí tras cada respuesta._",
                label="Fuentes recuperadas",
                elem_id="sources",
            )

    # ── Ejemplos ─────────────────────────────────────────────────────────────
    gr.Markdown("#### 💡 Preguntas de ejemplo")
    gr.Examples(
        examples=[[e] for e in EXAMPLES],
        inputs=msg_box,
        label="",
    )

    # ── Eventos ──────────────────────────────────────────────────────────────
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, sources_box],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box, sources_box],
    )
    clear_btn.click(
        lambda: ([], "", "_Las fuentes utilizadas aparecerán aquí tras cada respuesta._"),
        outputs=[chatbot, msg_box, sources_box],
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="yellow",
            neutral_hue="slate",
        ),
    )
