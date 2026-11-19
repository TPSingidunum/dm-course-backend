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
    not_worth = ~too_low & ( scored["score"] <= 2)
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

# TODO: Create new segmenting functions or predictsion, and we are waching for overlaping users, which suggest they have the highest chance to buy something
# This is, or could be a Predictsion algorithm, that calculates the scores of the user
def score_target_customers(eligible: pd.DataFrame, limit: int) -> pd.DataFrame:
    ordered = eligible.sort_values(["score", "customer_id"], ascending=[False, True]).head(limit)
    return ordered.assign(rank=range(1, len(ordered) + 1))

# Rule based mareking segmentation, A Marketing domain information that can't always be expressed into a number form
def rule_target_customers(scored: pd.DataFrame, campaign, limit: int):
    matching = scored[(scored["frequency"] >= campaign["rule_min_frequency"]) & (scored["recency_days"] < campaign["rule_max_recency_days"])]
    ordered = matching.sort_values(["recency_days", "customer_id"], ascending=[False, True]).head(limit)
    return ordered.assign(rank=range(1, len(ordered) + 1))

# Audience is a cross section of overlaping user in all different groups of potential customers
def audience_totals(audience: pd.DataFrame, campaign) -> dict:
    scored = audience[audience["score"].notna()]
    cost = round(len(audience) * campaign["contact_cost"], 4) 
    expected_net_value = float(scored["expected_value"].sum())
    return {
        "contacts": len(audience),
        "cost": cost,
        "expected_purchases": round(float(scored["probability"].sum()), 4), # scored probility of purcah 0 to 1
        "expected_net_value": expected_net_value,
        "expected_profit": round(expected_net_value - cost, 2),
        "unscored": len(audience) - len(scored) # These are the users which we did not market to
    }

# Could make it take N lists of users
def compare_customers(score_customers: pd.DataFrame, rule_customers: pd.DataFrame, campaign):
    score_customers_ids = set(score_customers["customer_id"].tolist())
    rule_customers_ids = set(rule_customers["customer_id"].tolist())
    both, either = score_customers_ids & rule_customers_ids, score_customers_ids | rule_customers_ids  # Inner join select, and outjoin select

    return {
        "score": {
            "customer_ids": list(score_customers_ids),
            "totals": audience_totals(score_customers, campaign)
        },
        "rule": {
            "customer_ids": list(rule_customers_ids),
            "totals": audience_totals(rule_customers, campaign)
        },
        "overlap": {
            "both": sorted(both),
            "only_score": sorted(score_customers_ids - rule_customers_ids),
            "only_rule": sorted(rule_customers_ids - score_customers_ids),
        }
    }