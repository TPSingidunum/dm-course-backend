
from dm_course_backend.models.conversion import build_training_dataset, split_dataset, train_covnversion_model, FEATURES
from dm_course_backend.models.data import load_events, load_orders
import pandas as pd

events, orders = load_events(), load_orders() 
dataset = build_training_dataset(events, orders, 7)
train, test = split_dataset(dataset)
model = train_covnversion_model(train, "random_forest")
if model == None:
    print("Error")
    exit()

weights = pd.Series(model[-1].feature_importances_ , index=FEATURES)
print(weights.round(2).sort_values())



