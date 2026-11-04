
import os
from pathlib import Path
import joblib

from data import load_events, load_orders
from segments import build_features, scale, summarize, train_kmeans

MODEL_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[0] / "model"))

def train(k: int=4) -> dict:
    events, orders = load_events(), load_orders
    now = max(events["timestamp"].max(), orders["timestamp"].max())
    features = build_features(events, orders, now)
    X, scaler = scale(features)
    model, score = train_kmeans(X, k=4)
    summary = {
        "k": k,
        "silhouette": score,
        "n_customers": len(features),
        "reference": now,
        "segments": summarize(features, model.labels_)
    }

    assignments = {int(uid): f"segment_{label}" for uid, label in zip(features, model.labels_)}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "summary": summary, "assignments": assignments}, MODEL_DIR / "weights.joblib" )
    return summary