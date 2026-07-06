"""
LexRAG — Gradio Space
=====================
Minimal, beautiful chat interface for LexRAG Legal Intelligence Terminal.

Built by Evolucent AI (https://evolucentai.com)
Engineered by Eulogik (https://eulogik.com)
"""
import os
import sys
import json
import random
import subprocess
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr

# ── Attempt to load live engine; fall back to demo mode ─────────────────────
LIVE_MODE = False
query_rag = None

try:
    from api.rag_engine import query_rag
    from api.utils import detect_jurisdiction
    LIVE_MODE = True
except Exception:
    pass

# ── Demo responses for when live mode is unavailable ────────────────────────
DEMO_RESPONSES = {
    "gst": (
        "**Goods and Services Tax (GST) in India**\n\n"
        "The GST is a comprehensive indirect tax levied on the supply of goods and services "
        "in India, effective from July 1, 2017. It subsumed multiple central and state taxes "
        "into a unified structure.\n\n"
        "**Key Rates:**\n"
        "| Rate | Category |\n"
        "|------|----------|\n"
        "| 5% | Essential goods |\n"
        "| 12% | Standard goods |\n"
        "| 18% | Most services |\n"
        "| 28% | Luxury & sin goods |\n\n"
        "**Jurisdiction:** India\n\n"
        "> *This is a sample response. Connect a live LexRAG server for grounded legal research.*"
    ),
    "vat": (
        "**Value Added Tax (VAT) in the UAE**\n\n"
        "The UAE introduced VAT at a standard rate of **5%** on January 1, 2018, under "
        "Federal Decree-Law No. 8 of 2017. The Federal Tax Authority (FTA) administers "
        "the VAT framework.\n\n"
        "**Key Features:**\n"
        "- Standard rate: 5%\n"
        "- Zero-rated: Exports, international transport, healthcare, education\n"
        "- Exempt: Residential property, local passenger transport\n"
        "- Registration threshold: AED 375,000 (mandatory)\n\n"
        "**Jurisdiction:** UAE\n\n"
        "> *This is a sample response. Connect a live LexRAG server for grounded legal research.*"
    ),
    "corporate tax": (
        "**Corporate Tax in the UAE**\n\n"
        "The UAE implemented Federal Corporate Tax (CT) effective for financial years "
        "starting on or after June 1, 2023, under Federal Decree-Law No. 47 of 2022.\n\n"
        "**Key Rates:**\n"
        "| Taxable Income | Rate |\n"
        "|---------------|------|\n"
        "| Up to AED 375,000 | 0% |\n"
        "| Above AED 375,000 | 9% |\n"
        "| Qualifying Free Zone entities | 0% (subject to conditions) |\n\n"
        "**Jurisdiction:** UAE\n\n"
        "> *This is a sample response. Connect a live LexRAG server for grounded legal research.*"
    ),
    "default": (
        "**LexRAG Legal Intelligence Terminal**\n\n"
        "I can help you research:\n\n"
        "- **Indian Law**: GST, Income Tax, TDS, Companies Act, SEBI, RBI regulations\n"
        "- **UAE Law**: VAT, Corporate Tax, Free Zone regulations, FTA rulings\n"
        "- **Comparative**: Cross-jurisdiction tax analysis, treaty interpretation\n\n"
        "**Try asking:**\n"
        "- \"What is the GST rate on restaurants?\"\n"
        "- \"UAE VAT on financial services\"\n"
        "- \"Compare GST and VAT frameworks\"\n"
        "- \"TDS under Section 194B\"\n\n"
        "> ⚡ **Built by [Evolucent AI](https://evolucentai.com)** — "
        "Engineered by [Eulogik](https://eulogik.com)\n"
        "> 🔗 [GitHub](https://github.com/eulogik/LexRAG) • "
        "[PyPI](https://pypi.org/project/lexrag/) • "
        "[Docs](https://github.com/eulogik/LexRAG#readme)"
    )
}

def get_demo_response(question):
    q = question.lower()
    for keyword in ["gst", "vat", "corporate tax", "corporate tax"]:
        if keyword in q:
            return DEMO_RESPONSES[keyword]
    return DEMO_RESPONSES["default"]

def format_sources(sources):
    if not sources:
        return ""
    lines = ["\n\n**Sources:**"]
    for s in sources[:5]:
        title = s.get("title", s.get("source", "Source"))
        jur = s.get("jurisdiction", "")
        url = s.get("url", "")
        score = s.get("score", 0)
        jur_tag = f" `{jur}`" if jur else ""
        link = f" [↗]({url})" if url else ""
        lines.append(f"- {title}{jur_tag} *(score: {score:.2f})*{link}")
    return "\n".join(lines)

def get_jurisdiction_badge(jur):
    icons = {"India": "🇮🇳", "UAE": "🇦🇪", "Both": "🌐", "General": "🌐"}
    icon = icons.get(jur, "🌐")
    return f"{icon} `{jur}`"

def chat_fn(message, history):
    if LIVE_MODE:
        try:
            result = query_rag(message)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            jurisdiction = result.get("jurisdiction", "Both")
            confidence = result.get("confidence", "GROUNDED")
            badge = get_jurisdiction_badge(jurisdiction)
            src_text = format_sources(sources)
            return f"{badge}\n\n{answer}{src_text}"
        except Exception as e:
            return f"{get_demo_response(message)}\n\n*(Live query failed: {e})*"
    else:
        return get_demo_response(message)

# ── Custom CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --primary: #1a1a2e;
    --secondary: #16213e;
    --accent: #e94560;
    --text: #eaeaea;
    --text-muted: #888;
    --border: #2a2a4a;
}
.gr-box { border-radius: 12px !important; }
.gr-text-input, .gr-textarea { 
    background: var(--secondary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
button.primary {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
footer { display: none !important; }
"""

# ── Build Interface ─────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Outfit"),
    ),
    title="LexRAG — Legal Intelligence Terminal",
    css=CUSTOM_CSS,
) as demo:

    gr.HTML("""
        <div style="
            text-align: center;
            padding: 2rem 1rem 0.5rem;
        ">
            <h1 style="
                font-family: 'Outfit', sans-serif;
                font-weight: 700;
                font-size: 2.2rem;
                letter-spacing: -0.04em;
                margin: 0;
            ">⚖️ LexRAG</h1>
            <p style="
                font-family: 'Outfit', sans-serif;
                font-weight: 300;
                color: #888;
                margin: 0.25rem 0 0 0;
                font-size: 0.9rem;
            ">
                Legal Intelligence Terminal —
                <strong>UAE &amp; Indian Law, Taxation &amp; Compliance</strong>
            </p>
        </div>
    """)

    chatbot = gr.Chatbot(
        height=500,
        bubble_full_width=False,
        show_label=False,
        avatar_images=(None, "⚖️"),
    )

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Ask about law, tax, or regulations...",
            container=False,
            scale=7,
        )
        send = gr.Button(
            "Send",
            variant="primary",
            scale=1,
            min_width=100,
        )

    gr.HTML("""
        <div style="
            text-align: center;
            padding: 1rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.75rem;
            color: #555;
        ">
            Built by <a href="https://evolucentai.com" 
                style="color:#888;text-decoration:underline;text-decoration-style:dotted;">
                Evolucent AI</a> —
            Engineered by <a href="https://eulogik.com"
                style="color:#888;text-decoration:underline;text-decoration-style:dotted;">
                Eulogik</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/eulogik/LexRAG"
                style="color:#555;text-decoration:underline;text-decoration-style:dotted;">
                GitHub</a>
            &nbsp;·&nbsp;
            <a href="https://pypi.org/project/lexrag/"
                style="color:#555;text-decoration:underline;text-decoration-style:dotted;">
                PyPI</a>
        </div>
    """)

    # ── Event Handlers ──────────────────────────────────────────────────────
    def respond(message, chat_history):
        if not message or not message.strip():
            return "", chat_history
        chat_history = chat_history or []
        chat_history.append((message, None))
        yield "", chat_history
        response = chat_fn(message, chat_history)
        chat_history[-1] = (message, response)
        yield "", chat_history

    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# ── Launch ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
