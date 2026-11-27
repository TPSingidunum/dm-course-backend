# Routes for the new-look dm-course-frontend (product page, homepage, shop admin)
# Thin wrappers around the Day 2-7 services: they only change the response shape to
# {"recommendations": [...], "fallback": bool, "fallback_reason": str | None, "method": str}
# that the frontend reads (types/Ml.ts RecList), and add the small pieces that were missing.

import os
import pandas as pd

from dm_course_backend.models.data import load_customers, load_events, load_orders, load_products
from dm_course_backend.models.recommendation import POPULARITY_WEIGHTS, recommend_popular_products, recommend_same_category
from dm_course_backend.services.recommendation_service import content_recommendations, mixed_recommendations
from dm_course_backend.services.segment_services import load as load_segments
from dm_course_backend.services.conversion_services import get_customer_probability, load as load_conversion

# Live events sent by the frontend (POST /events), kept in memory only, never written to a database or a file
live_events: list[dict] = []

COLUMNS = ["product_id", "slug", "score", "name", "description", "category_id", "price", "img_url"]


def rec_list(recs, method: str, fallback: bool = False, reason: str | None = None) -> dict:
    rows = recs.to_dict("records") if isinstance(recs, pd.DataFrame) else recs
    return {"recommendations": rows, "fallback": fallback, "fallback_reason": reason if fallback else None, "method": method}


def popular(limit: int, exclude=None) -> pd.DataFrame:
    return recommend_popular_products(load_products(), load_events(), limit=limit, exclude=exclude)


def product_exists(product_id: int) -> bool:
    return bool((load_products()["product_id"] == product_id).any())


# Product page: "Recommended Products" = his Day 5 mix (content + sponsors/affiliates/user rules)
def content_section(product_id: int, limit: int = 5, user_id: int | None = None) -> dict:
    if not product_exists(product_id):
        return rec_list(popular(limit), "popularity", True, "unknown_product")
    return rec_list(mixed_recommendations(product_id, limit=limit, user_id=user_id), "content_mix")


# Product page: "Customers Also Viewed" = products seen in the same sessions (co-view counts)
def collaborative_section(product_id: int, limit: int = 5) -> dict:
    events = load_events()
    ev = events[events["event_type"].isin(POPULARITY_WEIGHTS) & events["product_id"].notna()]
    sessions = ev.loc[ev["product_id"] == product_id, "user_session"].unique()
    others = ev[ev["user_session"].isin(sessions) & (ev["product_id"] != product_id)]
    if others.empty:
        return rec_list(popular(limit, exclude=[product_id]), "popularity", True, "no_interactions")
    counts = others.groupby(others["product_id"].astype(int))["user_session"].nunique().rename("score")
    df = load_products().merge(counts, left_on="product_id", right_index=True)
    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)
    df["score"] = df["score"].astype(float)
    return rec_list(df[COLUMNS], "co_view_sessions")


# Product page: "Semantically similar". Embeddings are not built yet,
# so we answer with same-category products (by popularity) and say so.
def semantic_section(product_id: int, limit: int = 5) -> dict:
    if not product_exists(product_id):
        return rec_list(popular(limit), "popularity", True, "unknown_product")
    recs = recommend_same_category(load_products(), load_events(), product_id, limit=limit)
    return rec_list(recs, "same_category", True, "no_embeddings_yet")


def store_event(event: dict) -> dict:
    live_events.append(event)
    del live_events[:-10000]  # keep only the last 10 000 events in memory
    return {"stored": True, "live_events": len(live_events)}


def session_product_ids(session_id: str) -> list[int]:
    # Most recent first: live events from the frontend, then the session's rows in events.csv
    live = [int(e["product_id"]) for e in reversed(live_events)
            if e.get("user_session") == session_id and e.get("product_id") and e.get("event_type") in POPULARITY_WEIGHTS]
    events = load_events()
    old = events[(events["user_session"] == session_id) & events["product_id"].notna() & events["event_type"].isin(POPULARITY_WEIGHTS)]
    old_ids = old.sort_values("timestamp", ascending=False)["product_id"].astype(int).tolist()
    return list(dict.fromkeys(live + old_ids))


# Product page: "Based on your session" = content recommendations for the last product
# this session touched, without the products it already saw
def session_section(session_id: str, limit: int = 5) -> dict:
    seen = session_product_ids(session_id)
    if not seen:
        return rec_list(popular(limit), "popularity", True, "not_enough_signals")
    recs = [r for r in content_recommendations(seen[0], limit + len(seen)) if r["product_id"] not in seen][:limit]
    return rec_list(recs, "session_content")


def product_platform(product_id: int) -> dict:
    row = load_products()[load_products()["product_id"] == product_id]
    if row.empty:
        return {"product_id": product_id, "ai_description": {"content": None, "source": "none", "module_class": "rule / template",
                                                             "fallback": True, "fallback_reason": "unknown_product"}}
    p = row.iloc[0]
    # A paid LLM call on every page view is too expensive, so the shop shows a template;
    # the AI text is generated on demand in the admin (POST /seo/generate/product, Day 7)
    return {"product_id": product_id, "ai_description": {
        "content": f"{p['name']} ({p['category_name']}): {p['description']}",
        "source": "template", "module_class": "rule / template", "fallback": True, "fallback_reason": "ai_on_demand_in_admin"}}


def rfm_bucket(user_id: int) -> dict:
    orders = load_orders()
    now = orders["timestamp"].max()
    per_user = orders.groupby("user_id").agg(last=("timestamp", "max"), frequency=("order_id", "nunique"), monetary=("total_price", "sum"))
    if user_id not in per_user.index:
        return {"bucket": None, "recency_days": None, "frequency": 0, "monetary": 0.0}
    per_user["recency_days"] = (now - per_user["last"]).dt.days
    # 1 (worst) .. 3 (best) per dimension, by rank among all buyers
    r = pd.qcut(-per_user["recency_days"].rank(method="first"), 3, labels=[1, 2, 3])
    f = pd.qcut(per_user["frequency"].rank(method="first"), 3, labels=[1, 2, 3])
    m = pd.qcut(per_user["monetary"].rank(method="first"), 3, labels=[1, 2, 3])
    u = per_user.loc[user_id]
    return {"bucket": f"R{r[user_id]}F{f[user_id]}M{m[user_id]}", "recency_days": int(u["recency_days"]),
            "frequency": int(u["frequency"]), "monetary": round(float(u["monetary"]), 2)}


def user_profile(user_id: int) -> dict:
    saved = load_segments()
    customers = load_customers()
    row = customers[customers["customer_id"] == user_id]
    if saved is not None:
        segment = {"label": saved["assignments"].get(user_id, "unsegmented"), "source": "kmeans"}
    elif not row.empty:
        segment = {"label": row.iloc[0]["segment"], "source": "customers.csv"}
    else:
        segment = {"label": "unsegmented", "source": "none"}
    conversion = get_customer_probability(user_id)
    if conversion is None:
        conversion = {"probability": None, "fallback": True, "reason": "untrained_model"}
    return {"user_id": user_id, "segment": segment, "rfm": rfm_bucket(user_id), "conversion": conversion}


def homepage(user_id: int, session_id: str | None = None, limit: int = 5) -> dict:
    profile = user_profile(user_id)
    seen = session_product_ids(session_id) if session_id else []
    if seen:
        recs = session_section(session_id, limit)
    else:
        events = load_events()
        mine = events[(events["user_id"] == user_id) & events["product_id"].notna()]
        if mine.empty:
            recs = rec_list(popular(limit), "popularity", True, "no_history")
        else:
            last = int(mine.sort_values("timestamp")["product_id"].iloc[-1])
            recs = rec_list(mixed_recommendations(last, limit=limit, user_id=user_id), "content_mix_last_viewed")
    label = profile["segment"]["label"]
    if recs["fallback"]:
        text = "Welcome! Today's most popular products."
    elif label != "unsegmented":
        text = f"Welcome back! Picks for {label} shoppers, based on what you viewed."
    else:
        text = "Welcome back! Picks based on what you viewed."
    hero = {"content": text, "source": "template", "module_class": "rule / template", "fallback": False, "fallback_reason": None}
    return {"user_id": user_id, "hero": hero, "profile": profile, "recommendations": recs}


def platform_modules() -> dict:
    key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    rows = [
        (2, "segments", "Customer segments", "machine learning (k-means)", load_segments() is not None, "GET /segments", "unsegmented"),
        (3, "conversion", "Purchase probability", "machine learning (classifier)", load_conversion() is not None, "GET /customers/{id}/conversion-probability", "base rate / none"),
        (4, "campaigns", "Campaign targeting", "ML + rules", True, "GET /campaigns/{id}/recommended-audience", "rule mode"),
        (5, "content", "Recommended products (mix)", "TF-IDF + rules", True, "GET /recommendations/content/{id}", "popular products"),
        (5, "collaborative", "Customers also viewed", "co-view counts", True, "GET /recommendations/collaborative/{id}", "popular products"),
        (5, "session", "Based on your session", "content of last viewed", True, "GET /recommendations/session/{session_id}", "popular products"),
        (5, "semantic", "Semantically similar", "same category (no embeddings yet)", False, "GET /recommendations/semantic/{id}", "same category"),
        (7, "seo", "AI SEO agent", "LLM via OpenRouter", key, "POST /seo/generate/product", "503 OPENROUTER_API_KEY not set"),
    ]
    cols = ["day", "id", "name", "class", "available", "endpoint", "fallback"]
    return {"modules": [dict(zip(cols, r)) for r in rows]}
