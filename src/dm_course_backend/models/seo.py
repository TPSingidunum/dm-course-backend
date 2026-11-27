# Day 7: AI SEO agent (new file)
# The model layer: everything here is plain Python + pandas, no network.
# We build the prompt, read the answer and check it. The service calls OpenRouter.

import json
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# Same limits as the SEO fields in the admin ProductForm.vue (zod schema)
TITLE_MAX = 60
DESCRIPTION_MAX = 155
KEYWORDS_COUNT = 5

SYSTEM_PROMPT = f"""You are an SEO expert for an online shop.
You write search-engine text for one product at a time.
Rules:
- "title": at most {TITLE_MAX} characters, include the product name.
- "description": the meta description, at most {DESCRIPTION_MAX} characters, one or two sentences.
- "keywords": exactly {KEYWORDS_COUNT} short search phrases, lowercase.
- "product_description": 2 to 3 sentences for the product page.
- Use only facts from the product data. Do not invent features.
Answer with one JSON object and nothing else:
{{"title": "...", "description": "...", "keywords": ["...", "..."], "product_description": "..."}}"""


class SEOContent(BaseModel):
    title: str = Field(min_length=3)
    description: str = Field(min_length=10)
    keywords: list[str] = Field(min_length=1)
    product_description: str = ""


def product_for_seo(products: pd.DataFrame, product_id: int) -> dict | None:
    row = products[products["product_id"] == product_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "title": row["name"],
        "description": row["description"],
        "category": row["category_name"],
        "price": float(row["price"]),
    }


def build_user_prompt(product: dict, prompt: str | None = None) -> str:
    lines = [
        f"Product name: {product['title']}",
        f"Category: {product.get('category') or 'unknown'}",
        f"Price: {product['price']} EUR" if product.get("price") is not None else "Price: unknown",
        f"Product description: {product['description']}",
    ]
    if prompt:
        # Extra instructions from the marketer (tone, audience, campaign)
        lines.append(f"Extra instructions: {prompt}")
    return "\n".join(lines)


def build_messages(product: dict, prompt: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(product, prompt)},
    ]


def parse_seo(text: str) -> SEOContent:
    # Models sometimes wrap JSON in ```json fences or add a sentence around it,
    # so we cut out the part between the first { and the last }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the answer")
    try:
        data = json.loads(text[start:end + 1])
        return SEOContent(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ValueError(f"invalid JSON: {e}") from e


def check_limits(seo: SEOContent) -> list[str]:
    problems = []
    if len(seo.title) > TITLE_MAX:
        problems.append(f"title is {len(seo.title)} characters, max is {TITLE_MAX}")
    if len(seo.description) > DESCRIPTION_MAX:
        problems.append(f"description is {len(seo.description)} characters, max is {DESCRIPTION_MAX}")
    if len(seo.keywords) != KEYWORDS_COUNT:
        problems.append(f"there are {len(seo.keywords)} keywords, we need {KEYWORDS_COUNT}")
    return problems


def trim_to_limits(seo: SEOContent) -> SEOContent:
    # Last resort: cut on a word boundary so the text still fits the form
    def cut(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        short = text[:limit + 1].rsplit(" ", 1)[0].rstrip(" ,.;:-")
        return short if 0 < len(short) <= limit else text[:limit]

    return SEOContent(
        title=cut(seo.title, TITLE_MAX),
        description=cut(seo.description, DESCRIPTION_MAX),
        keywords=seo.keywords[:KEYWORDS_COUNT],
        product_description=seo.product_description,
    )
