import os
from pathlib import Path
import pandas as pd
from functools import lru_cache

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[0] / "data"))

@lru_cache
def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "events.csv", parse_dates=["timestamp"])

@lru_cache
def load_orders() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["timestamp"])