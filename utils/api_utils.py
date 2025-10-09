# utils/api_utils.py
import os
import json
import time
import requests
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL, LLM_TEMPERATURE

HEADERS = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json",
}

def chat(system_prompt: str, user_prompt: str, max_tokens: int = 1200, temperature: float = LLM_TEMPERATURE):
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(3):
        try:
            res = requests.post(url, headers=HEADERS, json=data, timeout=60)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[retry {attempt+1}] {e}")
            time.sleep(1.0)
    raise RuntimeError("DeepSeek request failed after retries.")

def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = LLM_TEMPERATURE):
    prompt = user_prompt + "\n\nReturn ONLY valid JSON (no markdown, no commentary)."
    content = chat(system_prompt, prompt, max_tokens=max_tokens, temperature=temperature)
    try:
        start, end = content.find("{"), content.rfind("}")
        return json.loads(content[start:end+1])
    except Exception:
        raise ValueError(f"Invalid JSON response: {content[:300]}")
