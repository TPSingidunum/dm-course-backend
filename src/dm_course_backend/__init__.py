from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

import uvicorn
from dm_course_backend.services.segment_services import train, load
from dm_course_backend.services.campaign_service import build_audience, load as load_campaings, get_campaign
from dm_course_backend.models.targeting import compare_customers
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

class AudienceRow(BaseModel):
    rank: int
    customer_id: int
    segment: str
    probability: float | None
    value: float # what is the profit margin, or sales revenu
    expected_value: float # final profit (value - TCO)
    score: float

class AudienceResponse(BaseModel):
    rank: int
    customer_id: int
    segment: str
    probability: float | None
    value: float # what is the profit margin, or sales revenu
    expected_value: float # final profit (value - TCO)
    score: float


@app.get("/campaigns")
def get_campaigns():
    campaigns = load_campaings()
    print(campaigns)
    if campaigns is None:
        raise HTTPException(503, "No campaigns") # Error Codes, HTTP 400 Bad Request, ErrorCode 102 # 1 is Kmeans error, and 2 is not trained
    return campaigns.to_dict(orient='records');

@app.get("/campaigns/{campaign_id}/recommended-audience")
def recommended_audiences(campaign_id: int, mode: Literal["ml", "rule"] = "ml"):
    campaign = get_campaign(campaign_id)
    # ml mode is the score_customers
    score_customers, rule_customers, excluded = build_audience(campaign)
    result = compare_customers(score_customers, rule_customers, campaign)
    for side in ("score", "rule"):
        result[side]["totals"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in result[side]["totals"].items()}
    result["excluded"] = excluded
    result["mode"] = mode
    return result


# Campaign Section



def main() -> None:
    uvicorn.run(
        "dm_course_backend:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
