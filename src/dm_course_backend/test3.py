
from dm_course_backend.conversion import build_training_dataset, split_dataset, train_covnversion_model, FEATURES, predict_conversion_probaility, LABEL
from dm_course_backend.data import load_events, load_orders
import pandas as pd

events, orders = load_events(), load_orders() 
dataset = build_training_dataset(events, orders, 7)
train, test = split_dataset(dataset)
model = train_covnversion_model(train, "random_forest")

for user_id in [395, 1]:
    p = predict_conversion_probaility(model, dataset.loc[[user_id]])
    print("User ID:", user_id, "| Label:", p, " (True label:", dataset.loc[user_id, LABEL] ,")")

model2 = train_covnversion_model(train)

for user_id in [395, 1]:
    p = predict_conversion_probaility(model2, dataset.loc[[user_id]])
    print("User ID:", user_id, "| Label:", p, " (True label:", dataset.loc[user_id, LABEL] ,")")



