
from dm_course_backend.data import load_events, load_orders
from dm_course_backend.segments import purchase_summary, visit_summary, build_features, scale, train_kmeans, summarize
import pandas as pd

events, orders = load_events(), load_orders()
# print(len(events))
# print(purchase_summary(orders).loc[395])
# print(visit_summary(events).loc[395])


# Is in January 2026, some day in January
now = max(events["timestamp"].max(), orders["timestamp"].max())
features = build_features(events, orders, now)
print(features.shape)
print((features["frequency"] > 0 ).sum(), " order checkouts")
print(features.loc[395])
print(features)

X, scaler = scale(features)
print("Mean: ", X.mean(axis=0).round(2), "std:", X.std(axis=0).round(2))
print("User 395:", X[features.index.get_loc(395)].round(2))

model, score = train_kmeans(X, k=4)
print("Score:", score)
print("Segments:", pd.Series(model.labels_).value_counts().sort_index().to_dict())
print("User 395 Segment location:", model.labels_[features.index.get_loc(395)])

result = summarize(features, model.labels_)

df = pd.DataFrame(result)
print(df)

# Base on user Clasification
# Segment 1 is our Windows Shoppers
# Segment 0 is our Drfiting aways
# Segment 3 is our Regular Buyers 
# Segment 2 is our VIPs 
