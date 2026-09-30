"""Flask application for the fake news classifier in project.ipynb."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets"
app = Flask(__name__, template_folder=".", static_folder=None)
MODELS: dict[str, object] = {}
VECTORIZER: TfidfVectorizer | None = None
TRAINING_ERROR: str | None = None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", "", text)
    return re.sub(r"\s+", " ", text).strip()


def train_models() -> None:
    """Train the same four classifiers as the notebook from the two CSV files."""
    global MODELS, VECTORIZER, TRAINING_ERROR
    true_path, fake_path = DATA_DIR / "True.csv", DATA_DIR / "Fake.csv"
    if not true_path.exists() or not fake_path.exists():
        TRAINING_ERROR = (
            "Training data is missing. Put True.csv and Fake.csv in the Datasets "
            "folder next to app.py. Each CSV must contain a 'text' column."
        )
        return
    try:
        real = pd.read_csv(true_path)
        fake = pd.read_csv(fake_path)
        for label, frame, filename in ((1, real, "True.csv"), (0, fake, "Fake.csv")):
            if "text" not in frame.columns:
                raise ValueError(f"{filename} must contain a 'text' column.")
        data = pd.concat([real["text"], fake["text"]], ignore_index=True).fillna("")
        labels = pd.Series([1] * len(real) + [0] * len(fake))
        if len(data) < 4 or labels.nunique() != 2:
            raise ValueError("Both CSV files need examples; at least two rows per class are required.")

        cleaned = data.map(clean_text)
        x_train, _, y_train, _ = train_test_split(
            cleaned, labels, test_size=0.3, random_state=42, stratify=labels
        )
        vectorizer = TfidfVectorizer(max_features=100_000)
        train_vectors = vectorizer.fit_transform(x_train)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        for model in models.values():
            model.fit(train_vectors, y_train)
        MODELS, VECTORIZER, TRAINING_ERROR = models, vectorizer, None
    except Exception as exc:
        TRAINING_ERROR = f"Could not train models: {exc}"


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ready": bool(MODELS), "error": TRAINING_ERROR})


@app.post("/api/predict")
def predict():
    if not MODELS or VECTORIZER is None:
        return jsonify({"error": TRAINING_ERROR or "Models are not ready."}), 503
    payload = request.get_json(silent=True) or {}
    article = payload.get("news", "")
    if not isinstance(article, str) or not article.strip():
        return jsonify({"error": "Enter a news article to analyze."}), 400
    if len(article) > 100_000:
        return jsonify({"error": "Article is too long (maximum 100,000 characters)."}), 413

    vectors = VECTORIZER.transform([clean_text(article)])
    predictions = {
        name: ("Likely real" if int(model.predict(vectors)[0]) == 1 else "Likely fake")
        for name, model in MODELS.items()
    }
    real_votes = sum(label == "Likely real" for label in predictions.values())
    label = "Likely real" if real_votes >= (len(predictions) // 2 + 1) else "Likely fake"
    return jsonify({
        "predictions": predictions,
        "consensus": {"label": label, "votes": max(real_votes, len(predictions) - real_votes), "total": len(predictions)},
    })


train_models()

if __name__ == "__main__":
    if TRAINING_ERROR:
        print(f"Startup note: {TRAINING_ERROR}")
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop).")
    app.run(host="127.0.0.1", port=5000, debug=False)
