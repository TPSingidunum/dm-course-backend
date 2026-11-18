import pandas as pd

def score_customer(customers, campaign) -> dict:
    scored = customers.copy()
    # TCO, revenue, calcualte the margin profit per order
    scored["value"] = scored["expected_order_values"] * campaign["net_margin_rate"]
    scored["expected_value"] = scored["value"]*scored["probability"]
    
    # score isn't in the range of [0,1], and current it is in dollars
    scored["score"] = scored["expected_value"] - campaign["contact_cost"]

    return scored

def select_target_customer(scored: pd.DataFrame, campaign) -> tuple[pd.DataFrame, dict]:

    missing = scored["probability"].isna()
    too_low = scored["probability"] < campaign["min_probability"]
    # the money inversted is larged than the profit margin we get
    not_worth = ~too_low & ( scored["score"] <= 15)
    excluted = {
        "missing_count": int(missing.sum()),
        "too_low_count": int(too_low.sum()),
        "not_worth_count": int(not_worth.sum()),
    }
    return scored[~(missing | too_low | not_worth)], excluted

#TODO: Figure out what is happening here
def audience_limit(campaign) -> dict:
    # Every campaign runs n number of times 
    affordable = round(campaign["budget"] * 100) // round(campaign["contact_cost"] * 100 )
    limitContacts, reason = affordable, "budget"
    if campaign["max_contacts"] < limitContacts:
        return {
            "canStart": True,
            "limit": campaign["max_contacts"]
        }
    else:
        return {
            "canStart": False,
            "limit": None
        }

def rank_target_customers(eligible: pd.DataFrame, limit: int) -> pd.DataFrame:
    ordered = eligible.sort_values(["score", "customer_id"], ascending=[False, True]).head(limit)
    return ordered.assign(rank=range(1, len(ordered) + 1))

