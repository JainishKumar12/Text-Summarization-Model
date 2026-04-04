# 📝 Text Summarization using Fine-Tuned T5

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### 🚀 [**Try the Live Demo**](https://text-summarization-model-expgsxkw3bzlv9vmuhr2un.streamlit.app/)

*An end-to-end abstractive text summarization system powered by a fine-tuned T5 Transformer*

</div>

---

## 📸 Demo

<div align="center">

### 🏠 Home Interface
![Home UI](SCREEN-SHOTS/home.png)

### ✨ Generated Summary
![Result](SCREEN-SHOTS/results.png)

</div>

---

## 🧠 What is this project?

This project takes any long piece of text and generates a **concise, meaningful summary** using a fine-tuned **T5 (Text-to-Text Transfer Transformer)** model from Hugging Face. Unlike extractive summarization (which just picks sentences), this model **generates brand new sentences** that capture the core meaning — just like a human would.

Key highlights:
- 🔥 Fine-tuned T5 model using transfer learning
- ⚡ Two deployment options — FastAPI (local) and Streamlit (cloud)
- 🎨 Custom HTML/CSS frontend
- 🤗 Model hosted on Hugging Face Hub

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | PyTorch |
| NLP Model | Hugging Face Transformers (T5) |
| Local Backend | FastAPI + Uvicorn |
| Cloud Deployment | Streamlit |
| Model Hosting | Hugging Face Hub |
| Frontend | HTML / CSS |
| Training | Jupyter Notebook |

---

## 🧪 Model Details

- **Model:** T5 (Text-to-Text Transfer Transformer)
- **Architecture:** Encoder-Decoder (Sequence-to-Sequence)
- **Framework:** PyTorch
- **Task:** Abstractive Text Summarization
- **Hosted at:** [JainishKumar12/text-summarizer-t5](https://huggingface.co/JainishKumar12/text-summarizer-t5)

---

## ⚙️ How It Works

```
User Input (long text)
        ↓
  Text Cleaning & Preprocessing
        ↓
  T5 Tokenizer (max 512 tokens)
        ↓
  Fine-tuned T5 Model (Beam Search)
        ↓
  Generated Summary (max 150 tokens)
        ↓
   Displayed to User
```

---

## 📂 Project Structure

```text
TEXTSUMMARIZERAPP/
│
├── SCREEN-SHOTS/               # App screenshots
│   ├── home.png
│   └── results.png
├── templates/
│   └── index.html              # Custom HTML frontend
├── saved_summarization_model/  # Model weights (gitignored)
├── app.py                      # FastAPI backend (local)
├── streamlit_app.py            # Streamlit app (cloud)
├── summarizer.py               # Shared model logic
├── text_summarizer.ipynb       # Training notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run Locally

### Option 1 — FastAPI
```bash
git clone https://github.com/JainishKumar12/Text-Summarization-Model.git
cd Text-Summarization-Model
pip install -r requirements.txt
uvicorn app:app --reload
```
Open: `http://127.0.0.1:8000`

### Option 2 — Streamlit
```bash
git clone https://github.com/JainishKumar12/Text-Summarization-Model.git
cd Text-Summarization-Model
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Open: `http://localhost:8501`

---

## ☁️ Live Deployment

This app is deployed on **Streamlit Community Cloud** with the model hosted on **Hugging Face Hub**.

🔗 **Live App:** https://text-summarization-model-expgsxkw3bzlv9vmuhr2un.streamlit.app/

To deploy your own version:
1. Upload your model to Hugging Face Hub
2. Push your code to GitHub
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your repo and set main file as `streamlit_app.py`
5. Click **Deploy** 🎉

---

## ⚠️ Important Notes

- Dataset not included due to size constraints
- Model weights are hosted on Hugging Face Hub (not in this repo)
- You can fine-tune the model on your own dataset using `text_summarizer.ipynb`

---

## 🔮 Future Improvements

- [ ] Experiment with BART and Pegasus models
- [ ] Improve summarization quality with larger datasets
- [ ] Add support for PDF and document uploads
- [ ] Enhance UI/UX with more interactive features
- [ ] Add summary length control slider

---

## 📌 Learning Outcomes

- Understanding Transformer-based NLP models
- Fine-tuning pre-trained models for specific tasks
- Building end-to-end ML applications
- Deploying ML models to the cloud
- Integrating Hugging Face Hub for model hosting

---

## 👨‍💻 Author

**Jainish Kumar**
Aspiring AI/ML Engineer

[![GitHub](https://img.shields.io/badge/GitHub-JainishKumar12-181717?style=flat&logo=github)](https://github.com/JainishKumar12)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-JainishKumar12-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/JainishKumar12)

---

<div align="center">
⭐ If you found this project useful, consider giving it a star!
</div>