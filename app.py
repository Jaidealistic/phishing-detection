import streamlit as st
from transformers import pipeline
import torch

# ─── Page Setup ────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Detection", page_icon="🛡️", layout="centered")
st.title("Phishing Detection System")
st.write("Detect phishing in **Emails** and **URLs** using fine-tuned BERT and DistilBERT models.")

# ─── Load Saved Models ─────────────────────────────────────────
email_model_path = "jaisubi006/final_email_bert"
url_model_path   = "jaisubi006/final_url_distilbert"

device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_pipelines():
    email_clf = pipeline("text-classification", model=email_model_path, tokenizer=email_model_path, device=device)
    url_clf   = pipeline("text-classification", model=url_model_path, tokenizer=url_model_path, device=device)
    return email_clf, url_clf

email_clf, url_clf = load_pipelines()

# ─── Sidebar Selection ─────────────────────────────────────────
st.sidebar.header("Select Detection Type")
mode = st.sidebar.radio("", ["📧 Email Phishing Detection", "🌐 URL Phishing Detection"])

# ─── Email Detection ───────────────────────────────────────────
if mode.startswith("📧"):
    st.subheader("📧 Email Phishing Detection")
    email_text = st.text_area("Paste Email Text Below:", height=200)
    if st.button("Analyze Email"):
        if not email_text.strip():
            st.warning("Please enter some text.")
        else:
            result = email_clf(email_text)[0]
            label = result['label']
            score = result['score']
            color = "red" if "1" in label or "spam" in label.lower() or "phish" in label.lower() else "green"
            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.progress(score)
            st.caption(f"Confidence: {score:.2f}")

# ─── URL Detection ─────────────────────────────────────────────
else:
    st.subheader("🌐 URL Phishing Detection")
    url_text = st.text_input("Enter a URL:")
    if st.button("Check URL"):
        if not url_text.strip():
            st.warning("Please enter a URL.")
        else:
            result = url_clf(url_text)[0]
            label = result['label']
            score = result['score']
            color = "red" if "1" in label or "phish" in label.lower() else "green"
            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.progress(score)
            st.caption(f"Confidence: {score:.2f}")

st.divider()
st.caption("Built with BERT (Email) and DistilBERT (URL) — fine-tuned on phishing datasets.")
