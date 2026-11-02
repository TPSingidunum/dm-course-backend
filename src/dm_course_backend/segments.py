import pandas as pd

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



# 2. Scale the values proportionally

# 3. K-means (how many groups do we want to have n = 4)

# 4. Label each group by their defining caracteristic
