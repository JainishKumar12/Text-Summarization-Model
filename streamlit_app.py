import streamlit as st
from summarizer import clean_data, summarize_dialogue
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
        
        /* Hide sidebar and its toggle button */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

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

# ── Read input sent from HTML via query params ────────────────────────────────
user_input = st.query_params.get("text", "")
summary = ""
if user_input:
    with st.spinner("Summarizing..."):
        summary = summarize_dialogue(user_input, model, tokenizer, device)

# ── Load HTML and inject summary + fixed JS ───────────────────────────────────
with open("templates/index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Escape backticks and backslashes for safe JS injection
safe_input = user_input.replace("\\", "\\\\").replace("`", "\\`")
safe_summary = summary.replace("\\", "\\\\").replace("`", "\\`")

bridge_script = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
    const form = document.getElementById("summarization-form");
    const summaryText = document.getElementById("summary-text");
    const dialogueInput = document.getElementById("dialogue-input");
    const submitButton = form.querySelector("button");

    // Restore previous input and summary after reload
    const prevInput = `{safe_input}`;
    const prevSummary = `{safe_summary}`;

    if (prevInput) dialogueInput.value = prevInput;
    if (prevSummary) summaryText.innerText = prevSummary;

    form.addEventListener("submit", function(e) {{
        e.preventDefault();
        const dialogue = dialogueInput.value.trim();
        if (!dialogue) return;
        summaryText.innerText = "Processing...";
        submitButton.disabled = true;
        const encoded = encodeURIComponent(dialogue);
        window.location.href = "?text=" + encoded;
    }});
}});
</script>
"""

html = html.replace("</body>", bridge_script + "</body>")
st.components.v1.html(html, height=700, scrolling=False)