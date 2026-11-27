# Day 7: AI SEO agent (new file)
# The service layer: reads the key from the environment, calls OpenRouter,
# and runs the small agent loop: generate -> parse -> check -> ask to fix.

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import HTTPException

from dm_course_backend.models.data import load_products
from dm_course_backend.models.seo import (
    build_messages,
    check_limits,
    parse_seo,
    product_for_seo,
    trim_to_limits,
)

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_ATTEMPTS = 2  # first answer + one retry


def load_env_file(path: Path = ENV_FILE) -> None:
    # Tiny .env reader (KEY=value per line). A variable that is already set
    # in the real environment wins, so the .env file never overrides it.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def openrouter_settings() -> dict:
    load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "OPENROUTER_API_KEY not set")
    return {
        "api_key": api_key,
        "model": os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL,
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL,
    }


def call_openrouter(messages: list[dict], settings: dict) -> str:
    body = {
        "model": settings["model"],
        "messages": messages,
        "temperature": 0.5,  # same as the TinyLlama version: focused, a bit of variety
        "max_tokens": 500,
        "response_format": {"type": "json_object"},  # ask for JSON only (not every provider obeys)
        "reasoning": {"effort": "none"},  # DeepSeek V4.1 "thinks" by default; short SEO text does not need it
    }
    request = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
            "X-Title": "DM Course Backend",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise HTTPException(502, f"OpenRouter error {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(502, f"OpenRouter unreachable: {e}")
    return data["choices"][0]["message"]["content"] or ""


def generate_seo(product: dict, prompt: str | None = None) -> dict:
    settings = openrouter_settings()
    messages = build_messages(product, prompt)
    seo, problems = None, ["no answer yet"]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        answer = call_openrouter(messages, settings)
        try:
            seo = parse_seo(answer)
            problems = check_limits(seo)
        except ValueError as e:
            problems = [str(e)]
        if seo is not None and not problems:
            break
        # Agent step: give the model its own answer back and tell it what to fix
        fix = "Fix these problems and answer with the JSON object only: " + "; ".join(problems)
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": fix},
        ]

    if seo is None:
        raise HTTPException(502, "Model did not return valid JSON: " + "; ".join(problems))

    fallback = bool(problems)
    if fallback:
        seo = trim_to_limits(seo)

    return {
        **seo.model_dump(),
        "model": settings["model"],
        "attempts": attempt,
        "fallback": fallback,
        "problems": problems,
    }


def generate_product_seo(product_id: int, prompt: str | None = None) -> dict:
    product = product_for_seo(load_products(), product_id)
    if product is None:
        raise HTTPException(400, "Product does not exist")
    return {"product_id": product_id, **generate_seo(product, prompt)}
