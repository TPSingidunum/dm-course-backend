
# conversion metric for marketing is someone buying something
# How likelly is that a user will buy something - What we want to predict
# Statr to recomend to him to complete his actions of buying
# Send him ads to buy something, send him messagess to remind him
# sms, email, notification
# 0 - 1 where 0 is that user will not buy anything where 1 is that he will definitely buy
# New users will have a estimate of 0.5

# Targeting mechanism and budged
# 1. Targeting (which users do we target)
# 2. Budged (how much money will we spend on each segment)
# 3. Margin (What is the profit marging for each segment)
# 4. Timing (when to advertise to users)

# Timing, how oftern should we recommned users our products

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dm_course_backend.segments import build_features

import pandas as pd

FEATURES = ["recency_days", "frequency", "monetary", "aov", "conversion", "sessions", "cart_adds", "product_views"]

LABEL = "will_purchase"

def curstomer_features(events: pd.DataFrame, orders: pd.DataFrame, now) -> pd.DataFrame:
    features = build_features(events, orders, now)
    views = events[events["event_type"] == "product_view"]
    carts = events[events["event_type"] == "add_to_cart"]
    features["product_views"] = views.groupby("user_id").size()
    features["cart_adds"] = carts.groupby("user_id").size()
    features["sessions"] = events.groupby("user_id")["user_sessions"].nunique()

    df = features.fillna({
        "product_views": 0,
        "cart_adds": 0,
        "sessions": 0
    })

    return df[FEATURES]

def time_cutoff(events: pd.DataFrame, orders: pd.DataFrame, window_days: int = 7):
    now = max(events["timestamp"].max(), orders["timestamp"].max())
    return now - pd.Timedelta(days=window_days)

def build_training_dataset(events: pd.DataFrame, orders: pd.DataFrame, window_days: int = 7):
    cut_off = time_cutoff(events, orders, window_days)

    past_events = events[events["timestamp"] < cut_off]
    past_orders = orders[orders["timestamp"] < cut_off]

    dataset = curstomer_features(past_events, past_orders, now=cut_off)

    window_orders = orders[orders["timestamp"] >= cut_off]
    dataset[LABEL] = dataset.index.isin(window_orders["user_id"]).astype(int)

    return dataset

# We have a smaller buyer base than non buyer (12% percent)
def split_datset(dataset: pd.DataFrame, test_size: float = 0.3):
    # After split garante that some byuers are in both the test and train split 6%
    return train_test_split(dataset, test_size=test_size, random_state=42, stratify=dataset[LABEL])


def train_covnversion_model(train: pd.DataFrame, model_type: str = "logistic_regression"):
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        return None

    model = make_pipeline(StandardScaler(), classifier)
    return model.fit(train[FEATURES], train[LABEL])
