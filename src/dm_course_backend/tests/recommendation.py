# Recommendations
# Choose the product to sell to the people

# Metrics for product popularity
# 1. Affiliate links (Sponsors) Amazon choice 
# Affiliate link pays buy purchase
# Sponsor pays lump sum
# 2. Product popularity
# 3. User product popularity

# Popularity is based on events and their weights
# Calculate the popularity of a product 
# how many sales (add_to_cart events, purchase events)
# how many page views (view events)

# Events and Weights
# product_view = 1
# add_to_cart = 3
# purchase = 5

# Modifiers

# Sponsors or Affiliate products change the weights or events
# Sponsors 2 times multiplier to the weight value
# Affiliate 1.5 times multiplier

# Segments
# Per user segment, we need to calculate hte most popular products
# And apply modifiers to the event wights based on those values

# User based
# Products that are popular for the current user

# Natural Traffic, any other product (should not be completely random)

# User filters, for products he does not like

# Final recommendation logis, Need a global base modifier for each group modifiers we have


# Grading recommendations systems
# Based on user traffic

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
    calculate_weights_by_rules,
    mix_recommendations,
    popularity_from_events,
    prepare_product_text,
    recommend_popular_products,
    recommend_same_category,
    recommend_similar,
    score_products,
)
from dm_course_backend.services.recommendation_service import baseline_comparison, mixed_recommendations

products = load_products().copy()
events = load_events()
products["product_text"] = products["product_id"].map(prepare_product_text(products))
matrix, vec = build_tfidf_matrix(products)

print("Popular:", recommend_popular_products(products, events, limit=5)["product_id"].tolist())
print("Category(1):", recommend_same_category(products, events, 1, limit=5)["product_id"].tolist())
print("Content(1):", recommend_similar(products, 1, matrix, events, limit=5)[["product_id", "score"]].to_dict("records"))

rules = load_recommendation_config()
print("Weights[0]:", calculate_weights_by_rules(rules.iloc[0]))
scored = score_products(products, events, calculate_weights_by_rules(rules.iloc[0]), [101])
print("Scored user 101 top 3:", scored.sort_values(["score", "product_id"], ascending=[False, True]).head(3)["product_id"].tolist())

mixed = mixed_recommendations(1, limit=5)
print("Mixed(1):", [(r["product_id"], r["source"]) for r in mixed])
mixed_u = mixed_recommendations(1, limit=5, user_id=101)
print("Mixed(1, user=101):", [(r["product_id"], r["source"]) for r in mixed_u])
cmp = baseline_comparison(1, limit=5)
print("Overlap:", cmp["overlap"])
