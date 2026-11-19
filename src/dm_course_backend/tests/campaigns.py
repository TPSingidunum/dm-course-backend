# Digital Marketing cost money
# Both Ads and Recommendations cost money
# Grade our campaign success based on user predicted buyouts vs real case scenarios

# Budget of 10 000 eur
# SMS 0.4 EUR
# EMAIL 0.1 EUR 
# Flyers 0.5 eur
# Mail, 1 eur
# Google Ads 2500 eur
# Facebook Ads 2500 eur
# Coupons 5 eur
# etc. 1000

# For what segment of users do you run your campaigns

# VIP

# SMS 0.4 EUR
# EMAIL 0.1 EUR 
# Coupons 5 eur

# AOV = 200 eur
# Separate 200$ in budget to recommend for VIP user

# Window shoppers

# EMAIL 0.1 EUR 
# Flyers 0.5 eur
# Mail, 1 eur
# Coupons 5 eur

# AOV = 15$ 

# Drift Away customers

# EMAIL 0.1 EUR 
# Flyers 0.5 eur
# Mail, 1 eur
# Coupons 5 eur
import json

from dm_course_backend.models.data import load_campaigns, load_customers 
from dm_course_backend.models.targeting import audience_limit, compare_customers, rule_target_customers, score_customer, score_target_customers, select_target_customer

customers = load_customers() 
campaigns = load_campaigns()

scored = score_customer(customers, campaigns.loc[0])
print(scored)
selected_customers, nonselected_customers = select_target_customer(scored, campaigns.loc[0])
print(selected_customers)
print(nonselected_customers)

limit = audience_limit(campaigns.loc[0])
score_customers = score_target_customers(scored, limit)
rule_customers = rule_target_customers(scored, campaigns.loc[0], limit)

final_customers = compare_customers(score_customers, rule_customers, campaigns.loc[0])
print(json.dumps(final_customers, indent=4))
