# Product popularity

import math
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dm_course_backend.models.data import load_events

# Events and Weights
# product_view = 1
# add_to_cart = 3
# purchase = 5

POPULARITY_WEIGHTS = {
    "product_view": 1.0,
    "add_to_cart": 3.0,
    "purchase": 5.0
}

# Popularity vs recommendation score
# TODO: Do not hardcode the weights, make them configurable or prediciton based
def popularity_from_events(events: pd.DataFrame, weights=None) -> pd.Series:
    w = weights or POPULARITY_WEIGHTS
    ev = events[events["event_type"].isin(w)].copy()
    ev["product_id"] = pd.to_numeric(ev["product_id"], errors="coerce")
    ev = ev.dropna(subset=["product_id"]) # Drop evenets where product_id does not exits but category_id does
    ev["weight"] = ev["event_type"].map(w)
    return ev.groupby(ev["product_id"].astype(int))["weight"].sum()

# Part of our natural recommendation system, we need to recommend the most popular products to the user
def recommend_popular_products(products: pd.DataFrame, events: pd.DataFrame, limit=10, exclude: None = None) -> pd.DataFrame:
    pop = popularity_from_events(events)
    df = products.copy()
    df["score"] = df["product_id"].map(pop).fillna(0.0) if not pop.empty else 0.0

    # If these were natural recomendations, do not put sponsored or affiliate products in the list, as they are not natural recommendations
    if exclude is not None:
        df = df[~df["product_id"].isin(exclude)]

    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)

    columns = ["product_id", "slug", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

def recommend_same_category(products: pd.DataFrame, events: pd.DataFrame, product_id: int, limit=10, exclude: None = None) -> pd.DataFrame:
    extacted = products[products["product_id"] == product_id]
    if extacted.empty:
        raise ValueError(f"Unknown product {product_id}")
    category_id = extacted.iloc[0]["category_id"]

    df = products[(products["category_id"] == category_id) & (products["product_id"] != product_id)].copy()

    if exclude is not None:
        df = df[~df["product_id"].isin(exclude)]

    pop = popularity_from_events(events)
    df["score"] = df["product_id"].map(pop).fillna(0.0) if not pop.empty else 0.0
    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)
    
    columns = ["product_id", "slug", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

def prepare_product_text(products: pd.DataFrame) -> pd.Series:
    parts = products["name"].fillna("").astype(str) + " " + products["description"].fillna("").astype(str)
    if "category_name" in products.columns:
        parts = parts + " " + products["category_name"].fillna("").astype(str)

    return pd.Series(parts.str.strip().to_numpy(), index=products["product_id"].astype(int), name="product_text").str.lower()

def build_tfidf_matrix(products: pd.DataFrame, text_column: str = "product_text",
                       stop_words: str = "english", ngram_range: tuple[int, int] = (1, 2),
                       min_df: int = 2
                       ):
    texts = products[text_column] if text_column in products.columns else prepare_product_text(products)
    # Add custom weights for the TF-IDF vectorizer, we can use the popularity of the products as a weight for the TF-IDF vectorizer
    # For CPU devices, these fileds are good indicators of the popularity of the product, and can be used to weight the TF-IDF vectorizer
    vec = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range, min_df=min_df)
    tfidf_matrix = vec.fit_transform(list(texts.fillna("")))
    return tfidf_matrix, vec

def calculate_product_similarity(tfidf_matrix, item_id: int):
    return cosine_similarity(tfidf_matrix[item_id:item_id + 1], tfidf_matrix).flatten()

# Product simmularity
# Type 1 is by atribute values
# Type 2 by description similarity 
def recommend_similar(products: pd.DataFrame, product_id: int, tfidf_matrix, events: pd.DataFrame, limit=10, exclude: None = None) -> pd.DataFrame:
    # Prepare the text from comparison
    ids = products["product_id"].astype(int).to_numpy()
    pos = np.where(ids == product_id)[0]

    # TF-IDF vectorization of the product descriptions 
    # TF - Term Frequency - How many times a word appears in a document
    # IDF - Inverse Document Frequency - How important a word is across all documents

    if len(pos) == 0:
        return recommend_popular_products(products, events, limit=limit, exclude=exclude)

    i = int(pos[0])
    if tfidf_matrix[i].nnz == 0:
        return recommend_popular_products(products, events, limit=limit, exclude=exclude)

    similarities = calculate_product_similarity(tfidf_matrix, i)
    df = products.copy()
    df["score"] = np.round(similarities, 4)
    df = df[df["product_id"] != product_id]
    if exclude is not None:
        df = df[~df["product_id"].isin(exclude)]
    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)

    columns = ["product_id", "slug", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

# The final mix of paid / organic recommendations will be done in the recommendation.py file, where we will combine the results of the different recommendation methods and apply the modifiers for sponsored and affiliate products.
def calculate_weights_by_rules(rules: pd.Series, base_weighst=None):
    weights = dict(base_weighst or POPULARITY_WEIGHTS)
    features = rules["feature_modifier"] if "feature_modifier" in rules.index else None
    
    # Appling the feature modifiers for existing features
    if isinstance(features, list):
        for item in features:
            name = item["name"]
            weights[name] = weights.get(name, 1.0) * float(item["modifier"])

    base = float(rules["base_modifier"]) if pd.notna(rules.get("base_modifier")) else 1.0

    return {
        name: round(value * base, 10) for name, value in weights.items()
    }


# Combine populariy, rule weights, and nature recommnetsions 
# Score per rule type
def score_products(product: pd.DataFrame, events: pd.DataFrame, weights: dict, user_ids = None) -> pd.DataFrame:
    ev = events if user_ids is None else events[events["user_id"].isin(list(user_ids))]

    pop = popularity_from_events(ev, weights)
    out = product.copy()
    out["score"] = out["product_id"].map(pop).fillna(0.0)
    return out


# Is shown on a product page
def mix_recommendations(products: pd.DataFrame, events: pd.DataFrame, customers: pd.DataFrame, rule_config: pd.DataFrame, sponsors: pd.DataFrame, affiliates: pd.DataFrame, product_id: int, base_rec_content: pd.DataFrame, popular: pd.DataFrame, limit: int = 10, user_id: int | None = None):

    cols = ["product_id", "slug", "score", "name", "description", "category_id", "price", "img_url", "source"]

    rows = rule_config[["type", "total_split"]].copy()
    rows.loc[len(rows)] = ["Natural Traffic", max(0.0, 1.0 - float(rows["total_split"].sum()))]
    rows["raw_limit"] = (rows["total_split"] * limit).round(10)
    rows["slots"] = rows["raw_limit"].astype(int)
    # Rule recommendations, leftover slots go to the natural recommendations
    leftover = limit - int(rows["slots"].sum())
    if leftover > 0:
        rem = (rows["raw_limit"] - rows["slots"]).sort_values(ascending=False, kind="mergesort")
        rows.loc[rem.index[:leftover], "slots"] += 1

    # Scoring for each split type their products and returning them to our list
    slots = dict(zip(rows["type"], rows["slots"]))
    excluded = {product_id} if product_id is not None else set()
    # Parts or splits of product can overlap and i need ot keep an evidence list of used up product
    parts, used_content = [], False
    for id, row in rule_config.iterrows():
        split_type, n, w = row["type"], int(slots.get(row["type"], 0)), calculate_weights_by_rules(row)
        if n <= 0:
            continue

        scored = None # product groups by split
        if split_type == "User Based" and user_id is not None:
            scored = score_products(products, events, w, [user_id])
        elif split_type == "Sponsor Based":
            scored = score_products(products[products["product_id"].isin(sponsors["product_id"])], events, w)
        elif split_type == "Affiliate Based":
            scored = score_products(products[products["product_id"].isin(affiliates["product_id"])], events, w)
        elif split_type == "Segment Based":
            user_ids = customers["customer_id"]
            if user_id is not None:
                hit = customers[customers["customer_id"] == user_id]
                if not hit.empty:
                    user_ids = customers.loc[customers["segment"] == hit.iloc[0]["segment"], "customer_id"]
            scored = score_products(products, events, w, user_ids)

        # Edgde case scenarion
        if scored is not None:
            picked = scored[~scored["product_id"].isin(excluded) & (scored["score"] > 0)]
            picked = picked.sort_values(["score", "product_id"], ascending=[False, True]).head(n).copy()

            if not picked.empty:
                picked["source"] = split_type
                parts.append(picked[cols])
                excluded.update(picked["product_id"].tolist())
                # If n is 0, we have filled up the limit with all products
                # if n is > 0 we have elements left to fill
                n -= len(picked)
                # check it here and just add the rest of the products to be the natural recommendations

        if split_type == "User Based" and n > 0 and base_rec_content is not None and not base_rec_content.empty:
            picked = base_rec_content[~base_rec_content["product_id"].isin(excluded)].head(n).copy()
            if not picked.empty:
                picked["source"] = split_type
                parts.append(picked[cols])
                excluded.update(picked["product_id"].tolist())
                used_content = True

    # When we fill up the recommendation splits
    # Rest of the natural reccomendation
    natural = popular if used_content or base_rec_content is None or base_rec_content.empty else base_rec_content

    for frame in (natural, popular):
        left = limit - sum(len(p) for p in parts)
        if left > 0 and frame is not None and not frame.empty:
            picked = frame[~frame["product_id"].isin(excluded)].head(left).copy()
            if not picked.empty:
                picked["source"] = "Natural Traffic"
                parts.append(picked[cols])
                excluded.update(picked["product_id"].tolist())

    return pd.concat(parts, ignore_index=True).head(limit) if parts else products.iloc[0:0].copy()
