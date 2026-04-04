import streamlit as st
from summarizer import summarize_dialogue
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp { background-color: #050A18; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "JainishKumar12/text-summarizer-t5"

@st.cache_resource(show_spinner="Loading model, please wait...")
def load_model():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# ── Read and inject HTML ──────────────────────────────────────────────────────
with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# ── Get input from query params (bridge from HTML) ────────────────────────────
params = st.query_params
user_input = params.get("text", "")
summary = ""

if user_input:
    with st.spinner("Summarizing..."):
        summary = summarize_dialogue(user_input, model, tokenizer, device)

# ── Inject summary + JS bridge into HTML ─────────────────────────────────────
bridge_script = f"""
<script>
// Override the fetch call to use query params instead
document.addEventListener("DOMContentLoaded", function() {{
    const form = document.getElementById("summarization-form");
    const summaryText = document.getElementById("summary-text");

    // If summary already exists (returned from Python), show it
    const existingSummary = `{summary}`;
    if (existingSummary) {{
        summaryText.innerText = existingSummary;
    }}

    form.addEventListener("submit", function(e) {{
        e.preventDefault();
        const dialogue = document.getElementById("dialogue-input").value.trim();
        if (!dialogue) return;
        summaryText.innerText = "Processing...";
        // Redirect with text as query param so Streamlit can process it
        const encoded = encodeURIComponent(dialogue);
        window.location.href = "?text=" + encoded;
    }});
}});
</script>
"""

html = html.replace("</body>", bridge_script + "</body>")
st.components.v1.html(html, height=700, scrolling=False)