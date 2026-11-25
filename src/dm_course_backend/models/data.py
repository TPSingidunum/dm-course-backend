import os
from pathlib import Path
import pandas as pd
from functools import lru_cache

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

@lru_cache
def load_events() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "events.csv", parse_dates=["timestamp"])

@lru_cache
def load_orders() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["timestamp"])

@lru_cache
def load_customers() -> pd.DataFrame:
    print(DATA_DIR / "customers.csv")
    return pd.read_csv(DATA_DIR / "customers.csv")

@lru_cache
def load_campaigns() -> pd.DataFrame:
    return pd.read_json(DATA_DIR / "campaigns.json")

@lru_cache
def load_products() -> pd.DataFrame:
    products = pd.read_csv(DATA_DIR / "products.csv")
    categories = pd.read_csv(DATA_DIR / "categories.csv").rename(
        columns={"name": "category_name", "slug": "category_slug"}
    )
    return products.merge(categories[["category_id", "category_name"]], on="category_id", how="left")

@lru_cache
def load_sponsors() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sponsors.csv")

@lru_cache
def load_affiliates() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "affiliates.csv")

@lru_cache
def load_recommendation_config() -> pd.DataFrame:
    return pd.read_json(DATA_DIR / "recommendation.json")