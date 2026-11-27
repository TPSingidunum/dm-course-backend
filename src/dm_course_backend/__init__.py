from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

import uvicorn
from dm_course_backend.services.segment_services import train, load
from dm_course_backend.services.campaign_service import build_audience, get_audience_totals, load as load_campaigns, get_campaign, rounded
from dm_course_backend.models.targeting import audience_limit, compare_customers
from dm_course_backend.services.conversion_services import get_customer_probability, train as conversionTrain, load as conversionLoad
from dm_course_backend.services.recommendation_service import baseline_comparison, mixed_recommendations
from dm_course_backend.services.seo_service import generate_seo, generate_product_seo
from dm_course_backend.services.platform_service import (
    collaborative_section, content_section, homepage, platform_modules, product_platform,
    semantic_section, session_section, store_event, user_profile,
)

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



# Recommendation Section

class RecommendationItem(BaseModel):
    product_id: int
    slug: str
    score: float
    name: str
    description: str
    category_id: int
    price: float
    img_url: str
    source: str | None = None

@app.get("/recommendations/{product_id}", response_model=list[RecommendationItem], response_model_exclude_none=True)
def recommended_content(product_id: int, limit: int = 5, user_id: int | None = None):
    return mixed_recommendations(product_id, limit=limit, user_id=user_id)

@app.get("/recommendations/{product_id}/comparison")
def recommended_content_comparison(product_id: int, limit: int = 5):
    return baseline_comparison(product_id, limit=limit)

# Recommendation Section

# SEO Section (Day 7)

class SEORequest(BaseModel):
    title: str = Field(min_length=3)
    description: str = Field(min_length=10)
    category: str | None = None
    price: float | None = None
    prompt: str | None = None

class SEOPromptRequest(BaseModel):
    prompt: str | None = None

class SEOResponse(BaseModel):
    product_id: int | None = None
    title: str
    description: str
    keywords: list[str]
    product_description: str
    model: str
    attempts: int
    fallback: bool
    problems: list[str]

@app.post("/seo/generate/product", response_model=SEOResponse, response_model_exclude_none=True)
def seo_generate_product(body: SEORequest):
    return generate_seo(body.model_dump(), body.prompt)

@app.post("/seo/products/{product_id}", response_model=SEOResponse)
def seo_for_product(product_id: int, body: SEOPromptRequest = SEOPromptRequest()):
    return generate_product_seo(product_id, body.prompt)

# SEO Section

# Frontend Section (routes the new-look dm-course-frontend calls)

class AiSeoRequest(BaseModel):
    product_name: str = Field(min_length=3)
    description: str = Field(min_length=10)
    category_name: str | None = None

class EventIn(BaseModel):
    model_config = {"extra": "allow"}
    event_type: str
    user_id: int | None = None
    user_session: str | None = None
    product_id: int | None = None
    category_id: int | None = None
    recommendation_source: str | None = None
    recommendation_position: int | None = None

@app.get("/recommendations/content/{product_id}")
def recommendations_content(product_id: int, limit: int = 5, user_id: int | None = None):
    return content_section(product_id, limit, user_id)

@app.get("/recommendations/collaborative/{product_id}")
def recommendations_collaborative(product_id: int, limit: int = 5):
    return collaborative_section(product_id, limit)

@app.get("/recommendations/semantic/{product_id}")
def recommendations_semantic(product_id: int, limit: int = 5):
    return semantic_section(product_id, limit)

@app.get("/recommendations/session/{session_id}")
def recommendations_session(session_id: str, limit: int = 5):
    return session_section(session_id, limit)

@app.post("/events", status_code=201)
def post_event(body: EventIn):
    return store_event(body.model_dump())

@app.post("/recommendations/events", status_code=201)
def post_recommendation_event(body: EventIn):
    return store_event(body.model_dump())

@app.get("/platform/product/{product_id}")
def platform_product(product_id: int):
    return product_platform(product_id)

@app.get("/platform/profile/{user_id}")
def platform_profile(user_id: int):
    return user_profile(user_id)

@app.get("/platform/homepage")
def platform_homepage(user_id: int, session_id: str | None = None, limit: int = 5):
    return homepage(user_id, session_id, limit)

@app.get("/platform/modules")
def get_platform_modules():
    return platform_modules()

@app.post("/ai/seo")
def ai_seo(body: AiSeoRequest):
    seo = generate_seo({"title": body.product_name, "description": body.description, "category": body.category_name, "price": None})
    return {"seo_title": seo["title"], "seo_description": seo["description"], "keywords": seo["keywords"],
            "source": seo["model"], "fallback": seo["fallback"], "fallback_reason": "; ".join(seo["problems"]) or None}

# Frontend Section

def main() -> None:
    uvicorn.run(
        "dm_course_backend:app",
        host="127.0.0.1",
        port=3001,
        reload=True
    )
