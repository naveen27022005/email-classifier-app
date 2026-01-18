import streamlit as st
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Powered Email Classifier",
    layout="centered"
)

CATEGORY_REPO = "naveen-27022005/email-category-lr"
URGENCY_REPO = "naveen-27022005/email-urgency-lr"

CATEGORY_LABELS = ["complaint", "feedback", "other", "request", "spam"]
URGENCY_LABELS = ["low", "medium", "high"]

# ------------------------------------------------
# LOAD CATEGORY MODEL
# ------------------------------------------------
@st.cache_resource
def load_category_model():
    model_path = hf_hub_download(
        repo_id=CATEGORY_REPO,
        filename="category_lr.pkl"
    )
    vectorizer_path = hf_hub_download(
        repo_id=CATEGORY_REPO,
        filename="category_tfidf_vectorizer.pkl"
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ------------------------------------------------
# LOAD URGENCY MODEL
# ------------------------------------------------
@st.cache_resource
def load_urgency_model():
    model_path = hf_hub_download(
        repo_id=URGENCY_REPO,
        filename="urgency_lr.pkl"
    )
    vectorizer_path = hf_hub_download(
        repo_id=URGENCY_REPO,
        filename="urgency_tfidf_vectorizer.pkl"
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# ------------------------------------------------
# RULE-BASED FALLBACK
# ------------------------------------------------
def rule_based_urgency(text):
    text = text.lower()
    if any(w in text for w in ["urgent", "asap", "immediately", "down", "failure"]):
        return "high"
    if any(w in text for w in ["soon", "delay", "pending"]):
        return "medium"
    return "low"

# ------------------------------------------------
# PREDICTIONS
# ------------------------------------------------
def predict_category(text):
    model, vectorizer = load_category_model()
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    return CATEGORY_LABELS[idx], float(probs[idx])

def predict_urgency(text):
    model, vectorizer = load_urgency_model()
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)
    confidence = float(probs[idx])

    label = URGENCY_LABELS[idx]
    if confidence < 0.6:
        label = rule_based_urgency(text)

    return label, confidence

# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------
st.title("📧 AI Powered Smart Email Classifier")
st.caption("Lightweight, scalable ML system")

email_text = st.text_area("Enter email content")

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter email content")
    else:
        with st.spinner("Classifying email..."):
            category, cat_conf = predict_category(email_text)
            urgency, urg_conf = predict_urgency(email_text)

        st.success("Prediction Complete")
        st.markdown("### Results")
        st.write(f"**Category:** `{category}` (confidence: {cat_conf:.2f})")
        st.write(f"**Urgency:** `{urgency}` (confidence: {urg_conf:.2f})")
