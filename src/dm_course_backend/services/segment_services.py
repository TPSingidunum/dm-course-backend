
import os
from pathlib import Path
import joblib

from dm_course_backend.models.data import load_events, load_orders
from dm_course_backend.models.segments import build_features, scale, summarize, train_kmeans

MODEL_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[0] / "model"))

def train(k: int=4) -> dict:
    events, orders = load_events(), load_orders()
    now = max(events["timestamp"].max(), orders["timestamp"].max())
    features = build_features(events, orders, now)
    X, scaler = scale(features)
    model, score = train_kmeans(X, k)
    summary = {
        "k": k,
        "silhouette": score,
        "n_customers": len(features),
        "reference": now,
        "segments": summarize(features, model.labels_)
    }

    # Here we will include LLMs to read and process the segments, and appropriately name them
    # Will label the data with User friendly names

    assignments = {int(uid): f"segment_{label}" for uid, label in zip(features.index, model.labels_)}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "summary": summary, "assignments": assignments}, MODEL_DIR / "weights.joblib" )
    print(summary)
    return summary

def load() -> dict | None:
    return joblib.load(MODEL_DIR/"segments.joblib") if (MODEL_DIR / "segmenets.joblib").exists() else None