import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Segments - each segment represents a user group
# Rich customers, WindowShoppers, etc.  

# 1. Define what our features will be
# Recency - how long since last purchase
# Frequency - how often does a user order
# Monetary - how much money does a user spend
# Average Order Value
# Conversion - how much profit does each group give us.

FEATURES = ["recency_days", "frequency", "monetary", "aov", "conversion"]

def purchase_summary(orders: pd.DataFrame) -> pd.DataFrame:
    per_customer = orders.groupby("user_id")
    return pd.DataFrame({
        "frequency": per_customer["order_id"].nunique(),
        "monetary": per_customer["total_price"].sum(),
        "last_purchase" : per_customer["timestamp"].max()
    })


def visit_summary(events: pd.DataFrame) -> pd.DataFrame:
    per_customer = events.groupby("user_id")
    purchases = events[events["event_type"] == "purchase"]
    return pd.DataFrame({
        "sessions": per_customer["user_session"].nunique(),
        "buying_sessions": purchases.groupby("user_id")["user_session"].nunique(),
        "last_seen" : per_customer["timestamp"].max()
    })

def build_features(events: pd.DataFrame, orders: pd.DataFrame, now) -> pd.DataFrame:
    df = visit_summary(events).join(purchase_summary(orders), how="outer")
    df = df.fillna({"frequency": 0, "monetary": 0, "bying_sessions": 0})
    last_activity = df["last_purchase"].fillna(df["last_seen"])
    df["recency_days"] = (now - last_activity).dt.days
    df["monetary"] = df["monetary"].round(2)
    df["aov"] = (df["monetary"] / df["frequency"]).fillna(0).round(2)
    df["conversion"] = (df["buying_sessions"] / df["sessions"]).fillna(0).round(4)
    
    return df[FEATURES]


# 2. Scale the values proportionally
def scale(features: pd.DataFrame):
    scaler = StandardScaler().fit(features[FEATURES])
    return scaler.transform(features[FEATURES]), scaler

# 3. K-means (how many groups do we want to have n = 4)
def train_kmeans(X, k: int = 4):
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    score = silhouette_score(X, model.labels_)
    return model, round(float(score), 4)

# 4. Label each group by their defining caracteristic

def summarize(features: pd.DataFrame, labels) -> list[dict]:
    df = features.assign(segment=[ f"label{label}" for label in labels])
    table = df.groupby("segment")[FEATURES].mean().round(4)
    table.insert(0, "count", df["segment"].value_counts())
    table.insert(1, "share", (table["count"] / len(df)).round(4))
    overall = features.mean()
    table["profile"] = [describe(row, overall) for _, row in table.iterrows()]

    return table.reset_index().to_dict("records")


def describe(segment: pd.Series, overall: pd.Series) -> str:
    tags = []
    for col in FEATURES:
        if segment[col] > 1.25 * overall[col]:
            tags.append(f"{col}+")
        elif segment[col] < 0.75 * overall[col]:
            tags.append(f"{col}-")

    return " ".join(tags) or "average"