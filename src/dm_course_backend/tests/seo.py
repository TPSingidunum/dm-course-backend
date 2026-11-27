# Day 7: AI SEO agent checkpoint (new file)
# Runs without a key and without internet: we replace call_openrouter
# with a fake that returns answers we choose, so we can test our own code.

import json
from dm_course_backend.models.data import load_products
from dm_course_backend.models.seo import build_messages, check_limits, parse_seo, product_for_seo, trim_to_limits
from dm_course_backend.services import seo_service

products = load_products()
product = product_for_seo(products, 1)
print(product)

messages = build_messages(product, "Young audience, friendly tone")
print(messages[1]["content"])

# 1. A good answer, wrapped in ```json fences like many models do
good = {
    "title": "Smartphone Compact 686 | 5G OLED Phone, Black",
    "description": "Compact 5G smartphone with OLED display, dual camera and fast charging. Black, only 185.99 EUR.",
    "keywords": ["5g smartphone", "compact phone", "oled display", "dual camera phone", "fast charging phone"],
    "product_description": "A compact 5G phone with a bright OLED display and a dual camera. Fast charging gets you back online quickly.",
}
seo = parse_seo("```json\n" + json.dumps(good) + "\n```")
print(seo.title, len(seo.title), check_limits(seo))

# 2. A broken answer: parse_seo raises ValueError, the agent will retry
try:
    parse_seo('{"title": "Smartphone Compact 686", "keywords": "phone"}')
except ValueError as e:
    print("ValueError:", str(e).splitlines()[0])

# 3. Too long: check_limits finds the problems, trim_to_limits is the last resort
long_seo = seo.model_copy(update={"title": "Smartphone Compact 686 " * 4, "keywords": ["phone"]})
print(check_limits(long_seo))
print(trim_to_limits(long_seo).title)

# 4. The agent loop with a fake OpenRouter: bad JSON first, then a good answer
answers = iter(["Sure! Here is your SEO text: title = phone", json.dumps(good)])
seen = []

def fake_call(messages, settings):
    seen.append(len(messages))
    return next(answers)

seo_service.call_openrouter = fake_call
seo_service.openrouter_settings = lambda: {"api_key": "test", "model": "fake/model", "base_url": "http://fake"}
result = seo_service.generate_product_seo(1)
print(result["attempts"], result["fallback"], result["problems"], seen)
print(result["title"])
