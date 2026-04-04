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
        .block-container {
            max-width: 620px !important;
            padding-top: 50px !important;
        }
        /* Hide default streamlit textarea label */
        .stTextArea label { display: none !important; }
        /* Full width button */
        div.stButton { width: 100% !important; }
        div.stButton > button {
            width: 100% !important;
            background-color: #0EA5E9 !important;
            color: white !important;
            border: none !important;
            padding: 14px !important;
            font-size: 17px !important;
            border-radius: 5px !important;
        }
        div.stButton > button:hover {
            background-color: #0284C7 !important;
        }
        /* Textarea */
        .stTextArea textarea {
            background-color: #0D2137 !important;
            color: #C9D1E0 !important;
            border: 1px solid #1E3A5F !important;
            font-family: monospace !important;
            font-size: 15px !important;
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#0EA5E9;'>Text Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#0EA5E9; font-weight:700; margin-top:-15px;'>Using Hugging-Face Transformer</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#C9D1E0;'>Write or Paste your content below for quick summarization.</p>", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
dialogue = st.text_area("input", placeholder="Enter your content", height=150, label_visibility="collapsed")

# ── Button ────────────────────────────────────────────────────────────────────
clicked = st.button("Summarize")

# ── Summary box always visible ────────────────────────────────────────────────
summary_placeholder = st.empty()

if clicked:
    if dialogue.strip():
        with st.spinner("Processing..."):
            summary = summarize_dialogue(dialogue, model, tokenizer, device)
        summary_placeholder.markdown(f"""
            <div style="
                background:#0D2137;
                padding:15px 20px;
                border-radius:5px;
                border:1px solid #1E3A5F;
                margin-top:10px;
            ">
                <h4 style="color:#0EA5E9; text-align:center; margin-top:0;">Content Summary</h4>
                <p style="color:#C9D1E0; font-size:15px; line-height:1.6;">{summary}</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        summary_placeholder.warning("Please enter some content first.")
else:
    summary_placeholder.markdown("""
        <div style="
            background:#0D2137;
            padding:15px 20px;
            border-radius:5px;
            border:1px solid #1E3A5F;
            margin-top:10px;
        ">
            <h4 style="color:#0EA5E9; text-align:center; margin-top:0;">Content Summary</h4>
            <p style="color:#6B7A99; font-size:15px;">Your summary will appear here...</p>
        </div>
    """, unsafe_allow_html=True)