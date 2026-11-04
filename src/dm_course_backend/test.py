
from data import load_events, load_orders
from segments import purchase_summary, visit_summary, build_features, scale

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