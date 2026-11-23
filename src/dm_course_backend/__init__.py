from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

import uvicorn
from dm_course_backend.services.segment_services import train, load
from dm_course_backend.services.campaign_service import build_audience, get_audience_totals, load as load_campaigns, get_campaign, rounded
from dm_course_backend.models.targeting import audience_limit, compare_customers
from dm_course_backend.services.conversion_services import get_customer_probability, train as conversionTrain, load as conversionLoad

app = FastAPI(title="DM Backend")

@app.get("/health")
def health():
    return {
        "health": "ok"
    }

# Conversion Section

class ConversionTrainRequest(BaseModel):
    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    window_day: int = Field(7, ge=1, le=28)

@app.post("/models/conversion/train")
def train_conversion(body: ConversionTrainRequest = ConversionTrainRequest()):
    return conversionTrain(body.model_type, body.window_day) 

@app.get("/models/conversion/status")
def get_conversion_status():
    saved = conversionLoad()

    if saved is None:
        raise HTTPException(503, "untrained_model")

    return saved["meta"]

@app.get("/customers/{user_id}/conversion-probability")
def get_user_conversion_probability(user_id: int):
    result = get_customer_probability(user_id) 

    if result is None:
        raise HTTPException(503, "untrained_model")

    return result

# Conversion Section

# Segmentaion Section

class TrainRequest(BaseModel):
    k: int = Field(4, ge=2, le=10)

@app.post("/segments/train")
def train_segments(body: TrainRequest = TrainRequest()):
    return train(body.k)

@app.get("/segments")
def get_segments():
    saved_weights = load()
    if saved_weights is None:
        raise HTTPException(503, "untrained_model") # Error Codes, HTTP 400 Bad Request, ErrorCode 102 # 1 is Kmeans error, and 2 is not trained
    return saved_weights["summary"]

@app.get("/customers/{user_id}/segment")
def get_customer_segment(user_id: int):
    saved_weights = load()
    if saved_weights is None:
        raise HTTPException(503, "untrained_model")
    return {
        "user_id": user_id,
        "segment": saved_weights["assignments"].get(user_id, "unsegmented")
    } 

# Segmentaion Section

# Campaign Section

NOTE = ("Model-based expectations on illustrative data, same 7-day window as the probabilities. "
        "Not measured conversion lift and not incremental profit: that needs a randomized holdout.")

class Campaign(BaseModel):
    id: int
    name: str
    channel: str
    contact_cost: float
    net_margin_rate: float
    budget: float
    max_contacts: int
    min_probability: float
    window_days: int
    rule_max_recency_days: int
    rule_min_frequency: int

class AudienceRow(BaseModel):
    rank: int
    customer_id: int
    segment: str
    probability: float | None
    value: float # what is the profit margin, or sales revenu
    expected_value: float | None # final profit (value - TCO)
    score: float | None

class Totals(BaseModel):
    contacts: int
    cost: float
    expected_purchases: float
    expected_profit: float
    expected_net_value: float
    unscored: int

class AudienceResponse(BaseModel):
    campaign_id: int
    mode: Literal["ml", "rule"]
    limit_applied: int
    audience: list[AudienceRow]
    totals: Totals
    excluded: dict[str, int] | None

@app.get("/campaigns", response_model=list[Campaign])
def get_campaigns():
    campaigns = load_campaigns()
    if campaigns.empty:
        raise HTTPException(503, "No campaigns") # Error Codes, HTTP 400 Bad Request, ErrorCode 102 # 1 is Kmeans error, and 2 is not trained
    return campaigns.to_dict(orient='records');

@app.get("/campaigns/{campaign_id}/recommended-audience", response_model=AudienceResponse)
def recommended_audiences(campaign_id: int, mode: Literal["ml", "rule"] = "ml", limit: int | None = None):
    campaign = get_campaign(campaign_id)
    applied_limit = audience_limit(campaign) if limit is None else min(limit, audience_limit(campaign)) 
    # ml mode is the score_customers
    score_customers, rule_customers, excluded = build_audience(campaign, limit)

    # Returns results depending on the mode selected
    audience = score_customers if mode == "ml" else rule_customers
    totals =  {k: (v.item() if hasattr(v, "item") else v) for k, v in get_audience_totals(audience, campaign).items()}
    
    return {
        "campaign_id": campaign_id,
        "mode": mode,
        "limit_applied": applied_limit,
        "audience": rounded(audience),
        "totals": totals,
        "excluded": excluded if mode == "ml" else None
    }

@app.get("/campaigns/{campaign_id}/targeting-comparison")
def targeting_comparison(campaign_id: int, limit: int | None = None):
    campaign = get_campaign(campaign_id)
    applied_limit = audience_limit(campaign) if limit is None else min(limit, audience_limit(campaign))
    score_customers, rule_customers, excluded = build_audience(campaign, limit)

    result = compare_customers(score_customers, rule_customers, campaign)
    for mode in ["score", "rule"]:
        result[mode]["totals"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in result[mode]["totals"].items()}
    result.update(campaign_id=campaign_id, limit_applied=applied_limit, note=NOTE)
    return result

# Campaign Section



def main() -> None:
    uvicorn.run(
        "dm_course_backend:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
