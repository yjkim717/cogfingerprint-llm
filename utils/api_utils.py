import os
import json
import time
import re
import requests


# =========================================================================
# MULTI-MODEL CONFIGURATION
# =========================================================================
PROVIDER = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()

DEEPSEEK = {
    "base_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "model": "deepseek-chat",
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

GEMMA_4B = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("GEMMA_4B_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "google/gemma-3-4b-it",  
}

GEMMA_12B = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("GEMMA_12B_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "google/gemma-3-12b-it",
}

LLAMA_MAVRICK = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("LLAMA_MAVRICK_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "meta-llama/llama-4-maverick",
}

PROVIDERS = {
    "DEEPSEEK": DEEPSEEK,
    "GEMMA_4B": GEMMA_4B,
    "GEMMA_12B": GEMMA_12B,
    "LLAMA_MAVRICK": LLAMA_MAVRICK,
}

provider = PROVIDERS.get(PROVIDER, DEEPSEEK)

HEADERS = {
    "Authorization": f"Bearer {provider['api_key']}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/jennykim/cogfingerprint-llm",
    "X-Title": "CogFingerprint Generator",
}

LLM_MODEL = provider["model"]
LLM_BASE_URL = provider["base_url"]
LLM_TEMPERATURE = 0.7


# =========================================================================
# STABLE CHAT FUNCTION
# =========================================================================
def chat(system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = LLM_TEMPERATURE):
    """
    Robust chat completion with retries, exponential backoff, and detailed logging.
    """

    # Unified message schema
    if any(x in LLM_MODEL.lower() for x in ["gemma", "llama"]):
        merged_prompt = f"[System instruction]\n{system_prompt.strip()}\n\n[User]\n{user_prompt.strip()}"
        messages = [{"role": "user", "content": merged_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    data = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    endpoint = f"{LLM_BASE_URL}/chat/completions" if "openrouter.ai" in LLM_BASE_URL else LLM_BASE_URL

    print(f"\n[DEBUG] Sending to {LLM_MODEL} (max_tokens={max_tokens})")
    print(f"[DEBUG] Provider: {PROVIDER}, Endpoint: {endpoint}")

    # Retry logic with exponential backoff
    for attempt in range(1, 6):
        try:
            res = requests.post(endpoint, headers=HEADERS, json=data, timeout=180)

            # Meta moderation
            if res.status_code == 403 and "requires moderation" in res.text:
                print("[INFO] ⚠️ Meta moderation triggered — skipping this file.")
                return "[Llama 4 Maverick: Meta moderation flagged this input — output not generated.]"

            # Rate limit handling
            if res.status_code == 429:
                print(f"[WARN] Attempt {attempt}: 429 Rate limit hit → Retrying with backoff...")
                time.sleep(min(60, 2 ** attempt))
                continue

            # Other errors
            if res.status_code != 200:
                print(f"[WARN] Attempt {attempt}: Status {res.status_code} → {res.text[:150]}")
                raise requests.exceptions.HTTPError(f"Status {res.status_code}")

            # Successful
            result = res.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            wait_time = min(60, 2 ** attempt)
            print(f"[ERROR] Attempt {attempt} failed → {str(e)} | Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print(f"[FATAL] {PROVIDER} request failed after 5 attempts.")
    raise RuntimeError(f"{PROVIDER} request failed after retries.")


# =========================================================================
# JSON-ONLY WRAPPER (auto escape fix + empty/truncated response fallback)
# =========================================================================
def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = LLM_TEMPERATURE):
    """
    Wrapper forcing the model to return valid JSON output only.
    Cleans up formatting issues (```json, LaTeX, regex escapes, etc.)
    Includes automatic fallback for truncated or invalid JSON.
    """
    prompt = user_prompt + "\n\nReturn ONLY valid JSON (no markdown, no commentary)."
    content = chat(system_prompt, prompt, max_tokens=max_tokens, temperature=temperature)

    # Empty response handling
    if not content or not content.strip():
        print("[WARN] Empty response from model — skipping this file.")
        return {
            "keywords": [],
            "summary": "[Skipped due to empty response — no LLM output generated.]",
            "word_count": 0
        }

    # Cleanup output text
    cleaned = (
        content.replace("```json", "")
        .replace("```", "")
        .replace("\\'", "'")
        .replace("\\t", " ")
        .replace("\\n", " ")
        .strip()
    )

    # Meta moderation skip
    if "Meta moderation" in cleaned or "not generated" in cleaned:
        print("[INFO] Skipping JSON parsing due to Meta moderation flag.")
        return {
            "keywords": [],
            "summary": "[Skipped due to Meta moderation flag — no LLM output generated.]",
            "word_count": 0
        }

    #  Fix invalid escape characters
    cleaned_safe = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    cleaned_safe = re.sub(r'\\\\\\(?!["\\/bfnrtu])', r'\\\\\\\\', cleaned_safe)
    cleaned_safe = re.sub(r'"{', '{', cleaned_safe)
    cleaned_safe = re.sub(r'}"', '}', cleaned_safe)

    # Detect & fix truncated JSON
    start, end = cleaned_safe.find("{"), cleaned_safe.rfind("}")
    json_str = cleaned_safe[start:end + 1].strip()

    if json_str.count("{") > json_str.count("}"):
        print("[INFO] Detected truncated JSON → auto-fixing with closing brace.")
        json_str += "}"

    # Safe JSON parsing with retries
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSONDecodeError → {e}")
        print("Raw snippet:", cleaned_safe[:250])

        # Retry using unicode_escape
        try:
            json_str_retry = json_str.encode("unicode_escape").decode("utf-8")
            return json.loads(json_str_retry)
        except Exception:
            print("[FALLBACK] Could not parse JSON — returning default placeholder.")
            return {
                "keywords": [],
                "summary": "[Skipped due to invalid or truncated JSON response — parsing failed.]",
                "word_count": 0
            }
