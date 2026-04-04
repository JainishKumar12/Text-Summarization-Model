import streamlit as st
import streamlit.components.v1 as components
from summarizer import clean_data, summarize_dialogue
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit's default header/footer for a clean custom-UI look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container { padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ── Load model (cached so it only loads once) ─────────────────────────────────
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

# ── Load your custom HTML template ───────────────────────────────────────────
with open("templates/index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# ── Inject a hidden Streamlit bridge for receiving input from your HTML ───────
# We use a text_input widget hidden off-screen to pass the text from HTML → Python
user_input = st.text_input("__bridge_input__", key="bridge_input", label_visibility="hidden")

# ── Render your custom HTML ───────────────────────────────────────────────────
# The HTML is injected with a JS bridge so the Summarize button talks to Streamlit
html_with_bridge = html_content.replace(
    "</body>",
    """
    <script>
    // Bridge: when the user clicks Summarize in your HTML UI,
    // we push the text into Streamlit's hidden input widget.
    function summarizeFromHTML() {
        const textEl = document.getElementById("inputText");
        if (!textEl) return;
        const text = textEl.value.trim();
        if (!text) return;

        // Post message to Streamlit parent
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: text
        }, "*");
    }

    // Override the existing form submit / button click in your HTML
    document.addEventListener("DOMContentLoaded", function () {
        const btn = document.querySelector("button[type='submit'], #summarizeBtn, .btn-summarize");
        if (btn) {
            btn.addEventListener("click", function (e) {
                e.preventDefault();
                summarizeFromHTML();
            });
        }
    });
    </script>
    </body>"""
)

components.html(html_with_bridge, height=800, scrolling=True)

# ── Run summarization when input arrives ─────────────────────────────────────
if user_input:
    with st.spinner("Summarizing..."):
        summary = summarize_dialogue(user_input, model, tokenizer, device)

    # Display result below the custom UI
    st.markdown("---")
    st.subheader("📄 Summary")
    st.success(summary)

    # Optional: show a copy button
    st.code(summary, language=None)