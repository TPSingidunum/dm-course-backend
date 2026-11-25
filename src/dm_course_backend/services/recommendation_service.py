from fastapi import HTTPException

from dm_course_backend.models.data import (
    load_affiliates,
    load_customers,
    load_events,
    load_products,
    load_recommendation_config,
    load_sponsors,
)
from dm_course_backend.models.recommendation import (
    build_tfidf_matrix,
    mix_recommendations,
    prepare_product_text,
    recommend_popular_products,
    recommend_same_category,
    recommend_similar,
)


def content_recommendations(product_id: int, limit: int = 5) -> list[dict]:
    products = load_products().copy()
    events = load_events()
    products["product_text"] = products["product_id"].map(prepare_product_text(products))
    matrix, _ = build_tfidf_matrix(products)
    return recommend_similar(products, product_id, matrix, events, limit=limit).to_dict("records")


def mixed_recommendations(product_id: int, limit: int = 5, user_id: int | None = None) -> list[dict]:
    products = load_products().copy()
    events = load_events()
    if not (products["product_id"] == product_id).any():
        return recommend_popular_products(products, events, limit=limit).to_dict("records")
    products["product_text"] = products["product_id"].map(prepare_product_text(products))
    matrix, _ = build_tfidf_matrix(products)
    content = recommend_similar(products, product_id, matrix, events, limit=limit)
    popular = recommend_popular_products(products, events, limit=limit, exclude=[product_id])
    recs = mix_recommendations(
        products,
        events,
        load_customers(),
        load_recommendation_config(),
        load_sponsors(),
        load_affiliates(),
        product_id,
        base_rec_content=content,
        popular=popular,
        limit=limit,
        user_id=user_id,
    )
    return recs.to_dict("records")


def baseline_comparison(product_id: int, limit: int = 5) -> dict:
    products = load_products().copy()
    if not (products["product_id"] == product_id).any():
        raise HTTPException(400, "Product does not exist")
    events = load_events()
    content = content_recommendations(product_id, limit)
    category = recommend_same_category(products, events, product_id, limit=limit).to_dict("records")
    popular = recommend_popular_products(products, events, limit=limit, exclude=[product_id]).to_dict("records")
    mixed = mixed_recommendations(product_id, limit)
    c_ids = [r["product_id"] for r in content]
    k_ids = {r["product_id"] for r in category}
    p_ids = {r["product_id"] for r in popular}
    return {
        "product_id": product_id,
        "limit": limit,
        "content": content,
        "category": category,
        "popular": popular,
        "mixed": mixed,
        "overlap": {
            "content_vs_category": [i for i in c_ids if i in k_ids],
            "content_vs_popular": [i for i in c_ids if i in p_ids],
            "mixed_vs_content": [r["product_id"] for r in mixed if r["product_id"] in set(c_ids)],
            "mixed_vs_popular": [r["product_id"] for r in mixed if r["product_id"] in p_ids],
        },
    }
