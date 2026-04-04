import streamlit as st
from summarizer import summarize_dialogue
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")

# ── Match your dark blue design ──────────────────────────────────────────────
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp { background-color: #050A18; }

        .block-container {
            max-width: 620px !important;
            padding-top: 60px !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
            margin: 0 auto !important;
            background: #0A1628 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.15) !important;
        }

        h1 {
            color: #0EA5E9 !important;
            text-align: center !important;
            font-size: 2.2rem !important;
        }

        /* Subtitle - bold, similar size to title */
        h3 {
            color: #0EA5E9 !important;
            text-align: center !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            margin-top: -10px !important;
        }

        p, .stMarkdown p {
            color: #C9D1E0 !important;
        }

        /* Monospace textarea */
        .stTextArea textarea {
            background-color: #0D2137 !important;
            color: #C9D1E0 !important;
            border: 1px solid #1E3A5F !important;
            font-family: monospace !important;
            font-size: 15px !important;
            border-radius: 5px !important;
        }
        .stTextArea textarea:focus {
            border-color: #0EA5E9 !important;
            box-shadow: 0 0 8px rgba(14, 165, 233, 0.4) !important;
        }

        /* Full width tall button */
        .stButton { width: 100% !important; }
        .stButton > button {
            background-color: #0EA5E9 !important;
            color: white !important;
            border: none !important;
            width: 100% !important;
            font-size: 17px !important;
            padding: 14px !important;
            border-radius: 5px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: #0284C7 !important;
            box-shadow: 0 0 12px rgba(14, 165, 233, 0.5) !important;
        }

        /* Summary box with centered heading */
        .summary-box {
            background: #0D2137;
            padding: 15px 20px;
            border-radius: 5px;
            border: 1px solid #1E3A5F;
            color: #C9D1E0;
            margin-top: 10px;
        }
        .summary-box h4 {
            color: #0EA5E9 !important;
            text-align: center !important;
            margin-top: 0 !important;
            font-size: 1.2rem !important;
        }
        .summary-box p {
            color: #C9D1E0 !important;
            font-size: 15px !important;
            line-height: 1.6 !important;
        }
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

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>Text Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<h3>Using Hugging-Face Transformer</h3>", unsafe_allow_html=True)
st.markdown("<p>Write or Paste your content below for quick summarization.</p>", unsafe_allow_html=True)

dialogue = st.text_area("", placeholder="Enter your content", height=150, label_visibility="collapsed")

if st.button("Summarize"):
    if dialogue.strip():
        with st.spinner("Processing..."):
            summary = summarize_dialogue(dialogue, model, tokenizer, device)
        st.markdown(f"""
            <div class="summary-box">
                <h4>Content Summary</h4>
                <p>{summary}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some content first.")