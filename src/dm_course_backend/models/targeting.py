import pandas as pd

def score_customer(customers, campaign) -> dict:
    scored = customers.copy()
    # TCO, revenue, calculate the margin profit per order
    scored["value"] = scored["expected_order_value"] * campaign["net_margin_rate"]
    scored["expected_value"] = scored["value"]*scored["probability"]
    
    # score isn't in the range of [0,1], and currently it is in dollars
    scored["score"] = scored["expected_value"] - campaign["contact_cost"]

    return scored

def select_target_customer(scored: pd.DataFrame, campaign) -> tuple[pd.DataFrame, dict]:

    missing = scored["probability"].isna()
    too_low = scored["probability"] < campaign["min_probability"]
    # the money invested is larger than the profit margin we get
    not_worth = ~too_low & ( scored["score"] <= 15)
    excluded = {
        "missing_count": int(missing.sum()),
        "too_low_count": int(too_low.sum()),
        "not_worth_count": int(not_worth.sum()),
    }
    return scored[~(missing | too_low | not_worth)], excluded

def audience_limit(campaign) -> int:
    # contacts the budget pays for, in whole cents (2.00 // 0.40 would give 4.0)
    affordable = round(campaign["budget"] * 100) // round(campaign["contact_cost"] * 100)
    # the marketer's cap wins if it is smaller; 0 means the campaign can't start
    return int(min(affordable, campaign["max_contacts"]))

def rank_target_customers(eligible: pd.DataFrame, limit: int) -> pd.DataFrame:
    ordered = eligible.sort_values(["score", "customer_id"], ascending=[False, True]).head(limit)
    return ordered.assign(rank=range(1, len(ordered) + 1))

