
from fastapi import HTTPException

from dm_course_backend.models.data import load_campaigns, load_customers
from dm_course_backend.models.targeting import *

def load():
    return load_campaigns()

def get_campaign(campaign_id):
    campaigns = load()

    if campaign_id not in campaigns:
        raise HTTPException(400, "Campaign does not exist")

    return campaigns[campaign_id]

def build_audience(campaign):
    customers = load_customers()
    score = [score_customer(customer, campaign) for customer in customers ]
    a_limit = audience_limit(campaign)
    eligible, excluded = select_target_customer(score, campaign)
    score_customers = score_target_customers(eligible, a_limit)
    rule_customers = rule_target_customers(score, campaign, a_limit)

    return score_customers, rule_customers, excluded
