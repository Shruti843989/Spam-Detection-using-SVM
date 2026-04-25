import streamlit as st
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered",
)

# ── Load model & vectorizer ───────────────────────────────────────────────────
# Models live one level up (in the root of the repo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "svm_spam_model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

@st.cache_resource
def load_model():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

model, vectorizer = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📩 Spam Message Detector")
st.write("Uses a Support Vector Machine (SVM) trained on the SMS Spam Collection dataset.")

st.divider()

message = st.text_area(
    "Enter a message to classify:",
    placeholder="Type or paste your SMS / email message here…",
    height=160,
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned   = message.lower().strip()
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 **SPAM** — This message looks like spam.")
        else:
            st.success("✅ **HAM** — This message looks legitimate.")

st.divider()

# ── Batch check ───────────────────────────────────────────────────────────────
with st.expander("📋 Batch check (one message per line)"):
    batch_input = st.text_area(
        "Paste multiple messages, one per line:",
        height=200,
        key="batch",
    )
    if st.button("Check All", use_container_width=True):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if not lines:
            st.warning("No messages found.")
        else:
            vectorized_batch = vectorizer.transform([l.lower() for l in lines])
            preds = model.predict(vectorized_batch)
            for msg, pred in zip(lines, preds):
                label = "🚨 SPAM" if pred == 1 else "✅ HAM"
                st.write(f"**{label}** — {msg}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: SVM (linear kernel, C=1.0) · Vectorizer: TF-IDF (unigrams + bigrams, top-5000 features)")
