# Streamlit UI — Spam Detector

A simple web interface for the SVM-based spam detection model.

## How to run

```bash
# From the repo root
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

The app will open at `http://localhost:8501`.

## What it does

- **Single message check** — paste any SMS/email text and instantly classify it as Spam or Ham.
- **Batch check** — paste multiple messages (one per line) and classify them all at once.

The models (`svm_spam_model.pkl` and `tfidf_vectorizer.pkl`) are loaded from the repo root, so no paths need changing.
