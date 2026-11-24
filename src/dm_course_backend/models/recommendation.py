# Product popularity

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
    "product_view": 1,
    "add_to_cart": 3,
    "purchase": 5
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
    pop = popularity_from_events(events) if not pop.empty else None
    df = products.copy()
    df["score"] = df["product_id"].map(pop).fillna(0.0)

    # If these were natural recomendations, do not put sponsored or affiliate products in the list, as they are not natural recommendations
    if exclude is not None:
        df = df[~df["product_id"].isin(exclude)]

    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)

    columns = ["product_id", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

def recommend_same_category(products: pd.DataFrame, events: pd.DataFrame, product_id: int, limit=10, exclude: None = None) -> pd.DataFrame:
    extacted = products[products["product_id"] == product_id]
    category_id = extacted.iloc[0]["category_id"]

    df = products[(products["category_id"] == category_id) & (products["product_id"] != product_id)].copy()

    if exclude is not None:
        df = df[~df["product_id"].isin(exclude)]

    pop = popularity_from_events(events) if not pop.empty else None
    df["score"] = df["product_id"].map(pop).fillna(0.0)
    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)
    
    columns = ["product_id", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

def prepare_product_text(products: pd.DataFrame) -> pd.Series:
    parts = products["name"].fillna("").astype(str) + " " + products["description"].fillna("").astype(str)
    if "category_name" in products.columns:
        parts = parts + " " + products["category_name"].fillna("").astype(str)

    return pd.Series(parts.str.strip(), index=products["product_id"].astype(int), name="product_text").str.lower()

def build_tfidf_matrix(products: pd.DataFrame, text_column: str = "product_text",
                       stop_words: str = "English", ngram_range: tuple[int, int] = (1, 2),
                       min_df: int = 2
                       ):
    texts = products[text_column] if text_column in products.columns else prepare_product_text(products)
    # Add custom weights for the TF-IDF vectorizer, we can use the popularity of the products as a weight for the TF-IDF vectorizer
    # For CPU devices, these fileds are good indicators of the popularity of the product, and can be used to weight the TF-IDF vectorizer
    vec = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range, min_df=min_df)
    tfidf_matrix = vec.fit_transform(list(texts.fillna("")))
    return tfidf_matrix, vec

def calculate_product_similarity(tfidf_matrix, item_id: int) -> pd.Series:
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
    df = df.sort_values(["score", "product_id"], ascending=[False, True]).head(limit)

    columns = ["product_id", "score", "name", "description", "category_id", "price", "img_url"]
    return df[columns].reset_index(drop=True)

# The final mix of paid / organic recommendations will be done in the recommendation.py file, where we will combine the results of the different recommendation methods and apply the modifiers for sponsored and affiliate products.
def calculate_weights_by_rules(rules: pd.Series, base_weighst=None):
    weights = dict(POPULARITY_WEIGHTS or base_weighst)
    features = rules["feature_modifier"] if "feature_modifier" in rules.index else None
    
    # Appling the feature modifiers for existing features
    if isinstance(features, list):
        for item in features:
            name = item["name"]
            weights[name] = weights.get(name, 1.0) * float(item("modifier"))

    base = float(rules["base_modifier"]) if pd.notna(rules.get("base_modifier")) else 1.0

    return {
        name: round(value * base, 10) for name, value in weights.items()
    }


# Combine populariy, rule weights, and nature recommnetsions 
# Score per rule type
# def score_products():

# def mix_recommendatiosn():
    