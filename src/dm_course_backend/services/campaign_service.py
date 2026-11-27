
from fastapi import HTTPException

from dm_course_backend.models.data import load_campaigns, load_customers
from dm_course_backend.models.targeting import *

def load():
    return load_campaigns()

def get_campaign(campaign_id):
    campaigns = load()
    match = campaigns[campaigns["id"] == campaign_id]

    if match.empty:
        raise HTTPException(400, "Campaign does not exist")

    return match.iloc[0]

def build_audience(campaign, limit: int | None = None):
    customers = load_customers()
    score = score_customer(customers, campaign)
    a_limit = audience_limit(campaign)
    if limit is not None:
        a_limit = min(a_limit, limit)
    eligible, excluded = select_target_customer(score, campaign)
    score_customers = score_target_customers(eligible, a_limit)
    rule_customers = rule_target_customers(score, campaign, a_limit)

    return score_customers, rule_customers, excluded

def get_audience_totals(audience: pd.DataFrame, campaign) -> dict:
    return audience_totals(audience, campaign)

def rounded(audience: pd.DataFrame) -> list[dict]:
    rows = audience[["rank", "customer_id", "segment", "probability", "value", "expected_value", "score"]].to_dict("records")
    for row in rows:
        for key in ["value", "expected_value", "score"]:
            row[key] = None if pd.isna(row[key]) else round(float(row[key]), 2)
        row["probability"] = None if pd.isna(row["probability"]) else float(row["probability"])
    return rows

# Frontend shape (dm-course-frontend admin/campaigns page): extra fields only,
# the fields above (rank, customer_id, totals, ...) stay the same.

def rfm_buckets(customers: pd.DataFrame) -> pd.Series:
    # 1 (worst) .. 3 (best) per dimension, by rank among all customers
    r = pd.qcut(-customers["recency_days"].rank(method="first"), 3, labels=[1, 2, 3]).astype(str)
    f = pd.qcut(customers["frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(str)
    m = pd.qcut(customers["monetary"].rank(method="first"), 3, labels=[1, 2, 3]).astype(str)
    return pd.Series(("R" + r + "F" + f + "M" + m).to_numpy(), index=customers["customer_id"])

def reasons_for(row, campaign, mode: str) -> list[str]:
    if mode == "rule":
        n = int(row["frequency"])
        return [f"{n} order{'' if n == 1 else 's'} (min {int(campaign['rule_min_frequency'])})",
                f"last order {int(row['recency_days'])} days ago (< {int(campaign['rule_max_recency_days'])})"]
    return [f"probability {row['probability']:.0%} (min {campaign['min_probability']:.0%})",
            f"expected value {row['expected_value']:.2f} - cost {campaign['contact_cost']:.2f} = {row['score']:.2f}"]

def audience_view(audience: pd.DataFrame, rows: list[dict], campaign, mode: str, excluded: dict) -> dict:
    customers = load_customers()
    buckets = rfm_buckets(customers)
    by_id = audience.set_index("customer_id")
    for row in rows:
        c = by_id.loc[row["customer_id"]]
        row.update(user_id=row["customer_id"], rfm_bucket=buckets[row["customer_id"]],
                   recency_days=int(c["recency_days"]), frequency=int(c["frequency"]),
                   monetary=round(float(c["monetary"]), 2), reasons=reasons_for(c, campaign, mode))
    if mode == "ml":
        selected = len(customers) - sum(excluded.values())
    else:
        selected = int(((customers["frequency"] >= campaign["rule_min_frequency"]) & (customers["recency_days"] < campaign["rule_max_recency_days"])).sum())
    return {"campaign_name": campaign["name"], "total_candidates": len(customers), "selected": selected,
            "returned": len(rows), "fallback": False, "fallback_reason": None}

def comparison_metrics(score_customers: pd.DataFrame, rule_customers: pd.DataFrame, campaign) -> dict:
    def summary(audience):
        totals = audience_totals(audience, campaign)
        p = audience["probability"].dropna()
        return {"size": len(audience), "expected_conversions": round(float(totals["expected_purchases"]), 4),
                "expected_profit": round(float(totals["expected_profit"]), 2),
                "avg_probability": round(float(p.mean()), 4) if not p.empty else None}
    ml, rule = summary(score_customers), summary(rule_customers)
    a, b = set(score_customers["customer_id"]), set(rule_customers["customer_id"])
    return {"ml": ml, "rule": rule,
            "lift_expected_conversions": round(ml["expected_conversions"] / rule["expected_conversions"], 2) if rule["expected_conversions"] else None,
            "overlap": len(a & b), "jaccard": round(len(a & b) / len(a | b), 4) if a | b else None}
