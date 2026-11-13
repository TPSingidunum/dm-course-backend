import os
from pathlib import Path
import joblib

from dm_course_backend.conversion import _metrics
from dm_course_backend.data import load_events, load_orders
from dm_course_backend.conversion import *

MODEL_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[0] / "model"))

def train(model_type: str = "logistic_regression", window_days: int = 7):
    events, orders = load_events(), load_orders() 
    dataset = build_training_dataset(events, orders, window_days)
    train, test = split_dataset(dataset)
    model = train_covnversion_model(train, model_type)

    meta = {
        "model_type": model_type,
        "cutoff": str(time_cutoff(events, orders, window_days)),
        "n_train": len(train),
        "n_test": len(test),
        "base_rate": round(float(dataset[LABEL].mean())) ,# Is the relationship between our buyers and non buyers
        "metrics": _metrics(test[LABEL], model.predict_proba(test[FEATURES])),
    }

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "meta": meta
        },
        MODEL_DIR / "conversions.joblib"
    )

    return meta

def load() -> dict | None:
    return joblib.load(MODEL_DIR / "conversions.joblib") if (MODEL_DIR / "conversions.joblib").exists() else None