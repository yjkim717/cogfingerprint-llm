import os
import json
import time
import re
import requests


# =========================================================================
# MULTI-MODEL CONFIGURATION
# =========================================================================

DEEPSEEK_CONFIG = {
    "base_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "model": "deepseek-chat",
}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
FALLBACK_LOG_PATH = os.getenv("FALLBACK_LOG_PATH", os.path.join("logs", "fallback_samples.txt"))


def _append_fallback_record(sample: str, model: str, level: str):
    """Persist fallback-triggered samples for later reprocessing."""
    try:
        log_dir = os.path.dirname(FALLBACK_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp}\tmodel={model or 'UNKNOWN'}\tlevel={level or 'UNKNOWN'}\tsample={sample or 'UNKNOWN'}\n"
        with open(FALLBACK_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(line)
    except Exception as log_err:
        print(f"[FALLBACK-LOG][WARN] Could not record fallback sample: {log_err}")


def _extract_metadata_block(user_prompt: str) -> dict:
    """Parse HUMAN METADATA block for logging and diagnostics."""
    meta = {}
    try:
        block_match = re.search(r"HUMAN METADATA(.*?)HUMAN TEXT", user_prompt, re.DOTALL)
        if not block_match:
            return meta
        block = block_match.group(1)
        for raw_line in block.strip().splitlines():
            line = raw_line.strip()
            if not line.startswith("-"):
                continue
            if ":" not in line:
                continue
            key, value = line[1:].split(":", 1)
            meta[key.strip().lower()] = value.strip()
    except Exception:
        pass
    return meta

GEMMA_4B_CONFIG = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("GEMMA_4B_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "google/gemma-3-4b-it",
}

GEMMA_12B_CONFIG = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("GEMMA_12B_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "google/gemma-3-12b-it",
}

LLAMA_MAVRICK_CONFIG = {
    "base_url": OPENROUTER_BASE,
    "api_key": os.getenv("LLAMA_MAVRICK_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
    "model": "meta-llama/llama-4-maverick",
}

PROVIDERS = {
    "DEEPSEEK": DEEPSEEK_CONFIG,
    "GEMMA_4B": GEMMA_4B_CONFIG,
    "GEMMA_12B": GEMMA_12B_CONFIG,
    "LLAMA_MAVRICK": LLAMA_MAVRICK_CONFIG,
}

def get_provider():
    """Get provider configuration dynamically from environment variable."""
    provider_name = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
    return PROVIDERS.get(provider_name, DEEPSEEK_CONFIG), provider_name

def get_headers():
    """Get headers dynamically based on current provider."""
    provider, provider_name = get_provider()
    return {
        "Authorization": f"Bearer {provider['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/jennykim/cogfingerprint-llm",
        "X-Title": "CogFingerprint Generator",
    }

def get_llm_model():
    """Get LLM model dynamically based on current provider."""
    provider, _ = get_provider()
    return provider["model"]

def get_llm_base_url():
    """Get LLM base URL dynamically based on current provider."""
    provider, _ = get_provider()
    return provider["base_url"]

# For backward compatibility, keep these as properties that are computed dynamically
PROVIDER = os.getenv("LLM_PROVIDER", "DEEPSEEK").upper()
LLM_TEMPERATURE = 0.7


# =========================================================================
# STABLE CHAT FUNCTION
# =========================================================================
def chat(system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = 0.7):
    """
    Robust chat completion with retries, exponential backoff, and detailed logging.
    """
    # Get provider configuration dynamically
    provider, provider_name = get_provider()
    headers = get_headers()
    llm_model = get_llm_model()
    llm_base_url = get_llm_base_url()

    # Unified message schema
    if any(x in llm_model.lower() for x in ["gemma", "llama"]):
        merged_prompt = f"[System instruction]\n{system_prompt.strip()}\n\n[User]\n{user_prompt.strip()}"
        messages = [{"role": "user", "content": merged_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    data = {
        "model": llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    endpoint = f"{llm_base_url}/chat/completions" if "openrouter.ai" in llm_base_url else llm_base_url

    # Debug output is disabled by default to reduce noise in multi-threaded scenarios
    # Uncomment the following lines if you need to debug API calls:
    # print(f"\n[DEBUG] Sending to {llm_model} (max_tokens={max_tokens})")
    # print(f"[DEBUG] Provider: {provider_name}, Endpoint: {endpoint}")

    # Retry logic with exponential backoff
    for attempt in range(1, 6):
        try:
            res = requests.post(endpoint, headers=headers, json=data, timeout=180)

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

    print(f"[FATAL] {provider_name} request failed after 5 attempts.")
    raise RuntimeError(f"{provider_name} request failed after retries.")


# =========================================================================
# JSON-ONLY WRAPPER (auto escape fix + empty/truncated response fallback)
# =========================================================================
def chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = 0.7):
    """
    Wrapper forcing the model to return valid JSON output only for
    keyword/summary/word_count extraction.
    
    - Cleans up formatting issues (```json, LaTeX, regex escapes, etc.).
    - Attempts multiple fixes for truncated or malformed JSON.
    - FINAL FALLBACK (extreme quality): if JSON parsing still fails,
      calls the model once more with a minimal prompt to obtain a
      clean one-sentence summary, and returns that instead of a
      placeholder summary string.
    """
    prompt = user_prompt + "\n\nReturn ONLY valid JSON (no markdown, no commentary)."

    sample_meta = _extract_metadata_block(user_prompt)
    sample_meta_label = (sample_meta.get("identifier") or "").strip()
    if not sample_meta_label:
        fallback_parts = [
            (sample_meta.get("genre") or "").strip(),
            (sample_meta.get("subfield") or "").strip(),
            (sample_meta.get("year") or "").strip(),
        ]
        sample_meta_label = "_".join([part for part in fallback_parts if part]).strip("_")
        if not sample_meta_label:
            index_match = re.search(r"([A-Za-z]+_[A-Za-z0-9]+)_(\d{4})_(\d+)", user_prompt)
            if index_match:
                sample_meta_label = f"{index_match.group(1)}_{index_match.group(3)}"

    level_label = (sample_meta.get("level") or "").strip().upper()
    model_label = (sample_meta.get("model") or os.getenv("LLM_PROVIDER", "DEEPSEEK")).strip().upper()
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

    # Extract JSON object more carefully
    # Find the first complete JSON object (handle cases where there's extra text after)
    start = cleaned_safe.find("{")
    if start == -1:
        print("[WARN] No JSON object found in response.")
        return {
            "keywords": [],
            "summary": "[Skipped due to no JSON object found in response.]",
            "word_count": 0
        }
    
    # Try to find the matching closing brace
    brace_count = 0
    end = start
    for i in range(start, len(cleaned_safe)):
        if cleaned_safe[i] == '{':
            brace_count += 1
        elif cleaned_safe[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    # Extract the JSON string
    is_complete_json = (brace_count == 0)
    if is_complete_json:
        # Found complete JSON object - extract only this part (handles "Extra data" error)
        json_str = cleaned_safe[start:end + 1].strip()
    else:
        # Incomplete JSON - try to fix it (common Llama issue)
        meta_suffix = f" from {sample_meta_label}" if sample_meta_label else ""
        print(f"[INFO] Detected incomplete JSON{meta_suffix} → attempting to fix...")
        json_str = cleaned_safe[start:].strip()
        
        # Fix incomplete key-value pairs
        # If ends with a key without value (e.g., "word_count")
        if re.search(r'"word_count"\s*:?\s*$', json_str):
            json_str = re.sub(r'"word_count"\s*:?\s*$', '"word_count": 0', json_str)
        # If ends with a colon (e.g., "word_count":)
        elif json_str.rstrip().endswith(':'):
            # Remove trailing colon and add default value
            json_str = json_str.rstrip().rstrip(':').rstrip()
            if json_str.endswith('"word_count"'):
                json_str += ': 0'
            else:
                # Generic fix: remove last incomplete key-value pair
                json_str = re.sub(r',\s*"[^"]*"\s*:?\s*$', '', json_str)
                if not json_str.endswith('}'):
                    json_str += '}'
        # If missing closing brace
        if json_str.count("{") > json_str.count("}"):
            json_str += "}"
        # If ends with comma, remove it
        json_str = json_str.rstrip().rstrip(',').rstrip()
        if not json_str.endswith('}'):
            json_str += '}'
    
    # Safe JSON parsing with retries
    try:
        # Try parsing the extracted JSON
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        error_msg = str(e)
        print(f"[WARN] JSONDecodeError → {error_msg}")
        print("Raw snippet:", cleaned_safe[:300])
        print(f"Extracted JSON snippet: {json_str[:200]}...")
        
        # Check if error is "Extra data" (JSON is valid but followed by text)
        if "Extra data" in error_msg and is_complete_json:
            # JSON is complete, but there's extra text after it
            # We already extracted the complete JSON, so retry with just that part
            try:
                json_str_retry = cleaned_safe[start:end + 1].strip()
                result = json.loads(json_str_retry)
                print("[INFO] Successfully parsed JSON by extracting first complete object (Extra data fix).")
                return result
            except Exception:
                pass
        
        # Try fixing common issues: incomplete word_count
        try:
            # Fix incomplete word_count (most common Llama issue)
            if re.search(r'"word_count"\s*:?\s*$', json_str):
                # word_count key exists but no value
                json_str_fixed = re.sub(r'"word_count"\s*:?\s*$', '"word_count": 0', json_str)
                if json_str_fixed != json_str:
                    # Ensure closing brace
                    if json_str_fixed.count("{") > json_str_fixed.count("}"):
                        json_str_fixed += "}"
                    result = json.loads(json_str_fixed)
                    print("[INFO] Successfully parsed JSON after fixing incomplete word_count.")
                    return result
            elif re.search(r'"word_count"\s*:?\s*[,}]', json_str):
                # word_count exists but value is missing or incomplete
                json_str_fixed = re.sub(
                    r'"word_count"\s*:?\s*[,}]',
                    '"word_count": 0}',
                    json_str
                )
                if json_str_fixed != json_str:
                    result = json.loads(json_str_fixed)
                    print("[INFO] Successfully parsed JSON after fixing word_count value.")
                    return result
        except Exception:
            pass
        
        # Try regex extraction as last resort (for malformed JSON)
        try:
            # Find JSON-like pattern with required keys
            json_match = re.search(
                r'\{[^{}]*"keywords"[^{}]*"summary"[^{}]*"word_count"[^{}]*\}',
                cleaned_safe,
                re.DOTALL
            )
            if json_match:
                matched_str = json_match.group(0)
                # Fix incomplete word_count in matched string
                if re.search(r'"word_count"\s*:?\s*$', matched_str):
                    matched_str = re.sub(r'"word_count"\s*:?\s*$', '"word_count": 0', matched_str)
                if matched_str.count("{") > matched_str.count("}"):
                    matched_str += "}"
                result = json.loads(matched_str)
                print("[INFO] Successfully parsed JSON using regex extraction.")
                return result
        except Exception:
            pass
        
        # Final fallback: extreme quality path
        # Try to log which sample this corresponds to (based on HUMAN METADATA).
        meta_label_for_log = sample_meta_label
        
        if meta_label_for_log:
            print(f"[FALLBACK] Could not parse JSON — attempting high-quality summary fallback for sample: {meta_label_for_log}.")
        else:
            print("[FALLBACK] Could not parse JSON — attempting high-quality summary fallback.")
        
        _append_fallback_record(
            meta_label_for_log or "UNKNOWN_SAMPLE",
            model_label or "UNKNOWN_MODEL",
            level_label or "UNKNOWN_LEVEL",
        )
        
        # As a last resort, call the model again with a minimal prompt
        # that includes the original human text, to obtain a clean,
        # single-sentence summary. This avoids JSON formatting issues
        # and focuses only on getting a usable summary.
        try:
            # Try to recover the original human text from the user_prompt.
            # user_prompt is built in prompt_utils.generate_extraction_prompts as:
            #
            # HUMAN TEXT
            # \"\"\"{text}\"\"\"
            #
            text_match = re.search(r'HUMAN TEXT\\s+"""(.*)"""', user_prompt, re.DOTALL)
            original_text = text_match.group(1).strip() if text_match else ""

            fallback_user_prompt = (
                "You are reading a news article.\n"
                "TEXT:\n"
                f"\"\"\"{original_text}\"\"\"\n\n"
                "Write ONE English sentence that summarizes the main content of this article.\n"
                "The sentence must be at most 30 words.\n"
                "Do NOT talk about tasks, instructions, requests, or 'the text'.\n"
                "Do NOT explain what you are doing.\n"
                "Return ONLY the summary sentence as plain text."
            )
            fallback_summary = chat(
                system_prompt,
                fallback_user_prompt,
                max_tokens=120,
                temperature=0.3,
            )
            # Normalize fallback summary
            if not fallback_summary or not fallback_summary.strip():
                raise RuntimeError("Empty fallback summary.")
            
            # Best-effort cleanup: enforce single concise, content-focused sentence
            # 1) collapse whitespace
            fallback_summary_clean = " ".join(fallback_summary.strip().split())
            # 2) split into sentences by basic punctuation
            sentences = re.split(r'(?<=[.!?])\s+', fallback_summary_clean)
            # Prefer sentences that do NOT mention tasks/instructions
            def _is_tasky(s: str) -> bool:
                lowered = s.lower()
                task_keywords = [
                    "task is to",
                    "your task is",
                    "you are asked",
                    "instruction",
                    "please summarize",
                    "summarize the text",
                    "summarize this text",
                    "read the text",
                    "given text",
                    "provided text",
                    "the text does not",
                    "the original response",
                    "not applicable",
                    "no article to summarize",
                    "no article available",
                    "no passage to rewrite",
                    "there is no article",
                    "there is no text",
                    "response is not applicable",
                ]
                return any(k in lowered for k in task_keywords)

            candidate = ""
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                if not _is_tasky(s_clean):
                    candidate = s_clean
                    break

            if not candidate:
                # Fall back to first non-empty sentence, even if a bit task-like
                candidate = next((s for s in sentences if s.strip()), fallback_summary_clean)

            # If the chosen candidate still looks like a task/instruction sentence,
            # fall back to a simple rule-based summary from the original text.
            if _is_tasky(candidate):
                # Use the first sentence from original_text as a last-resort summary
                # to guarantee content-based information.
                original_text_clean = " ".join(original_text.split())
                orig_sentences = re.split(r'(?<=[.!?])\s+', original_text_clean)
                content_candidate = next((s for s in orig_sentences if s.strip()), original_text_clean)
                candidate = content_candidate or candidate

            # 3) enforce <= 30 words
            words = candidate.split()
            if len(words) > 30:
                candidate = " ".join(words[:30])
            fallback_summary_clean = candidate.strip()

            # Without reliable JSON, we cannot trust model word_count,
            # so set word_count to 0 here; downstream code may fall back
            # to naive length if needed.
            preview = fallback_summary_clean[:120]
            print(f"[FALLBACK] Using high-quality fallback summary: {preview!r}")
            return {
                "keywords": [],
                "summary": fallback_summary_clean,
                "word_count": 0,
            }
        except Exception as e2:
            print(f"[FALLBACK-ERROR] Failed to obtain fallback summary: {e2}")
            # Absolute last resort: placeholder summary
            return {
                "keywords": [],
                "summary": "[Skipped due to invalid or truncated JSON response — parsing failed.]",
                "word_count": 0,
            }
