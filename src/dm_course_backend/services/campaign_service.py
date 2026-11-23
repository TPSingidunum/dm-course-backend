
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
