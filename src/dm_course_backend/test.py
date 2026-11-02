
from data import load_events, load_orders
from segments import purchase_summary, visit_summary

events, orders = load_events(), load_orders()

print(len(events))

print(purchase_summary(orders).loc[395])
print(visit_summary(events).loc[395])