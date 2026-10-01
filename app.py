from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets"

app = Flask(__name__, template_folder=".", static_folder=None)

MODEL = None
VECTORIZER = None
TRAINING_ERROR = None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)
    return re.sub(r"\s+", " ", text).strip()


def train_model():
    global MODEL, VECTORIZER, TRAINING_ERROR

    true_path = DATA_DIR / "True.csv"
    fake_path = DATA_DIR / "Fake.csv"

    if not true_path.exists() or not fake_path.exists():
        TRAINING_ERROR = "True.csv and Fake.csv are missing."
        return

    try:
        real = pd.read_csv(true_path, usecols=["text"]).dropna()
        fake = pd.read_csv(fake_path, usecols=["text"]).dropna()

        real = real.sample(min(len(real), 10000), random_state=42)
        fake = fake.sample(min(len(fake), 10000), random_state=42)

        data = pd.concat(
            [real["text"], fake["text"]],
            ignore_index=True
        )

        labels = [1] * len(real) + [0] * len(fake)

        data = data.map(clean_text)

        x_train, _, y_train, _ = train_test_split(
            data,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        VECTORIZER = TfidfVectorizer(
            max_features=20000,
            stop_words="english"
        )

        x_train = VECTORIZER.fit_transform(x_train)

        MODEL = LogisticRegression(
            max_iter=300,
            random_state=42
        )

        MODEL.fit(x_train, y_train)

        del x_train
        del data
        del real
        del fake

    except Exception as e:
        TRAINING_ERROR = str(e)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ready": MODEL is not None,
        "error": TRAINING_ERROR
    })


@app.post("/api/predict")
def predict():
    if MODEL is None or VECTORIZER is None:
        return jsonify({
            "error": TRAINING_ERROR or "Model is not ready."
        }), 503

    payload = request.get_json(silent=True) or {}
    article = payload.get("news", "")

    if not isinstance(article, str) or not article.strip():
        return jsonify({
            "error": "Enter a news article to analyze."
        }), 400

    if len(article) > 100000:
        return jsonify({
            "error": "Article is too long."
        }), 413

    cleaned = clean_text(article)
    vector = VECTORIZER.transform([cleaned])

    prediction = MODEL.predict(vector)[0]

    label = "Likely real" if int(prediction) == 1 else "Likely fake"

    return jsonify({
        "predictions": {
            "Logistic Regression": label
        },
        "consensus": {
            "label": label,
            "votes": 1,
            "total": 1
        }
    })


train_model()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )