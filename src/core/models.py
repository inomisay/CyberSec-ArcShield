import requests
import json
import os
import re
import time


def _resolve_gemini_api_key(api_key=None):
    """Resolve Gemini API key from the environment."""
    return api_key or os.getenv("GEMINI_API_KEY")


DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_COMPLETION_TOKENS = 512
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_QUANTIZATION = "Q4_K_M"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_CLOUDFLARE_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
DEFAULT_GITHUB_MODEL = "Phi-4"

class ModelClient:
    """Base class for AI model clients."""
    def generate(self, prompt, system_prompt=None):
        raise NotImplementedError("Subclasses must implement generate()")

class OllamaClient(ModelClient):
    """Client for local Ollama instances."""
    def __init__(self, model_name=DEFAULT_OLLAMA_MODEL, base_url="http://localhost:11434", temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, top_k=40, max_tokens=DEFAULT_MAX_TOKENS):
        self.model_name = model_name
        self.url = f"{base_url}/api/generate"
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.top_k = top_k
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.think = self._resolve_thinking_mode(model_name)
        self.quantization = DEFAULT_OLLAMA_QUANTIZATION if model_name == DEFAULT_OLLAMA_MODEL else "unknown"

    @staticmethod
    def _resolve_thinking_mode(model_name):
        configured = os.getenv("OLLAMA_THINK")
        if configured is not None:
            return configured.strip().lower() in {"1", "true", "yes", "on"}
        lowered = str(model_name).lower()
        if "qwen3" in lowered or "deepseek-r1" in lowered:
            return False
        return None

    def generate(self, prompt, system_prompt=None):
        # Prepend system prompt for simple models or use 'system' field if supported
        final_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        
        data = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            }
        }
        if self.think is not None:
            data["think"] = self.think
        
        try:
            response = requests.post(self.url, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "") or result.get("thinking", "")
        except Exception as e:
            return f"Ollama Error: {e}"

class GeminiClient(ModelClient):
    """Client for Google Gemini API."""
    def __init__(self, api_key=None, model_name=DEFAULT_GEMINI_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, top_k=1, max_tokens=DEFAULT_MAX_TOKENS):
        self.api_key = _resolve_gemini_api_key(api_key)
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.top_k = top_k
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        # Masked debug
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] GeminiClient initialized (Model: {model_name}, Key: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "Gemini Error: API Key not found. Please add GEMINI_FLASH_LITE_API_KEY to your .env file."
            print(f"[!] {err}")
            return err
        
        headers = {'Content-Type': 'application/json'}
        
        # Prepare content
        data = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
            ,"generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        # Official system instruction field for Gemini 1.5
        if system_prompt:
            data["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Gemini ({self.model_name})...")
                response = requests.post(f"{self.url}?key={self.api_key}", json=data, headers=headers, timeout=90)

                # 429: Rate Limit
                if response.status_code == 429:
                    try:
                        error_payload = response.json()
                    except Exception:
                        error_payload = {}
                    error_msg = error_payload.get('error', {}).get('message', response.text or '')
                    retry_delay = None
                    for detail in error_payload.get('error', {}).get('details', []):
                        retry_delay = detail.get('retryDelay')
                        if retry_delay:
                            break
                    match = re.search(r'retry in ([\d.]+)s', error_msg, re.IGNORECASE)
                    delay_match = re.match(r'([\d.]+)s$', str(retry_delay or ''))
                    if delay_match:
                        wait = float(delay_match.group(1)) + 5
                    elif match:
                        wait = float(match.group(1)) + 5
                    else:
                        wait = 60 + (attempt * 30)
                    print(f"[!] Rate limited. Waiting {wait:.1f}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                    continue

                # 503: Service Unavailable / High Demand
                if response.status_code == 503:
                    print(f"[!] Server Busy (503). Waiting 30s before retry {attempt+1}/{max_retries}...")
                    time.sleep(30)
                    continue

                if response.status_code != 200:
                    json_err = response.json()
                    error_msg = json_err.get('error', {}).get('message', 'Unknown Error')
                    
                    # Some "high demand" errors might come as other status codes with specific messages
                    if "high demand" in error_msg.lower() or "overloaded" in error_msg.lower():
                        print(f"[!] {error_msg}. Waiting 45s before retry {attempt+1}/{max_retries}...")
                        time.sleep(45)
                        continue
                        
                    print(f"[!] Gemini API Error: {error_msg}")
                    return f"Gemini Error: {error_msg}"

                json_response = response.json()
                if 'candidates' in json_response and json_response['candidates']:
                    candidate = json_response['candidates'][0]
                    content = candidate.get('content') or {}
                    parts = content.get('parts') or []
                    if parts and parts[0].get('text'):
                        return parts[0]['text']
                    finish_reason = candidate.get("finishReason", "unknown")
                    safety_ratings = candidate.get("safetyRatings", [])
                    return (
                        "Gemini Safety Block: no response content returned "
                        f"(finish_reason={finish_reason}, safety_ratings={safety_ratings})"
                    )
                prompt_feedback = json_response.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "unknown")
                safety_ratings = prompt_feedback.get("safetyRatings", [])
                return (
                    "Gemini Safety Block: no response content returned "
                    f"(block_reason={block_reason}, safety_ratings={safety_ratings})"
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                print(f"[!] Gemini Connection Exception ({type(e).__name__}): {e}. Retrying {attempt+1}/{max_retries}...")
                time.sleep(10) # Short wait before connection retry
                continue
            except Exception as e:
                print(f"[!] Gemini Client Exception: {e}")
                return f"Gemini Error: {e}"

        return "Gemini Error: Max retries exceeded (api error or rate limit)."


class GroqClient(ModelClient):
    """Client for Groq API (OpenAI-compatible chat completions)."""

    def __init__(self, api_key=None, model_name=DEFAULT_GROQ_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] GroqClient initialized (Model: {model_name}, Key: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "Groq Error: API Key not found. Please add GROQ_API_KEY to your .env file."
            print(f"[!] {err}")
            return err

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
            # Keep responses compact to reduce TPM pressure during benchmark loops.
            "max_tokens": self.max_tokens,
        }

        max_retries = 12
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Groq ({self.model_name})...")
                response = requests.post(self.url, json=data, headers=headers, timeout=60)

                if response.status_code == 429:
                    try:
                        error_msg = response.json().get("error", {}).get("message", "")
                    except Exception:
                        error_msg = response.text or ""

                    match = re.search(r"try again in\s*([\d.]+)s", error_msg, re.IGNORECASE)
                    wait = float(match.group(1)) + 0.7 if match else (2.0 + attempt)
                    print(
                        f"[!] Groq rate limited. Waiting {wait:.2f}s before retry "
                        f"{attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    try:
                        error_msg = response.json().get("error", {}).get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    print(f"[!] Groq API Error: {error_msg}")
                    return f"Groq Error: {error_msg}"

                json_response = response.json()
                choices = json_response.get("choices", [])
                if choices and choices[0].get("message"):
                    return choices[0]["message"].get("content", "")
                return "Groq Error: No response content returned."
            except Exception as e:
                print(f"[!] Groq Client Exception: {e}")
                return f"Groq Error: {e}"

        return "Groq Error: Max retries exceeded (rate limit)."

class MistralClient(ModelClient):
    """Client for Mistral AI API (OpenAI-compatible chat completions)."""

    def __init__(self, api_key=None, model_name=DEFAULT_MISTRAL_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.url = "https://api.mistral.ai/v1/chat/completions"
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] MistralClient initialized (Model: {model_name}, Key: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "Mistral Error: API Key not found. Please add MISTRAL_API_KEY to your .env file."
            print(f"[!] {err}")
            return err

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Mistral ({self.model_name})...")
                response = requests.post(self.url, json=data, headers=headers, timeout=60)

                if response.status_code == 429:
                    # Mistral free tier has tight rate limits.
                    wait = 10.0 + (attempt * 5)
                    print(
                        f"[!] Mistral rate limited. Waiting {wait:.2f}s before retry "
                        f"{attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    try:
                        error_payload = response.json()
                        error_msg = (
                            error_payload.get("error", {}).get("message")
                            or error_payload.get("message")
                            or json.dumps(error_payload)
                        )
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    print(f"[!] Mistral API Error ({response.status_code}): {error_msg}")
                    return f"Mistral Error: {error_msg}"

                json_response = response.json()
                choices = json_response.get("choices", [])
                if choices and choices[0].get("message"):
                    return choices[0]["message"].get("content", "")
                return "Mistral Error: No response content returned."
            except Exception as e:
                print(f"[!] Mistral Client Exception: {e}")
                return f"Mistral Error: {e}"

        return "Mistral Error: Max retries exceeded (rate limit)."

class OpenAIClient(ModelClient):
    """Client for official OpenAI API."""

    def __init__(self, api_key=None, model_name=DEFAULT_OPENAI_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.max_completion_tokens = DEFAULT_MAX_COMPLETION_TOKENS
        self.url = "https://api.openai.com/v1/chat/completions"
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] OpenAIClient initialized (Model: {model_name}, Key: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "OpenAI Error: API Key not found. Please add OPENAI_API_KEY to your .env file."
            print(f"[!] {err}")
            return err

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        # Newer models prefer 'developer' role over 'system' or strictly only 'user'
        is_reasoning_model = self.model_name.startswith(("o1", "o3", "gpt-5"))
        system_role = "developer" if is_reasoning_model else "system"
        
        if system_prompt:
            messages.append({"role": system_role, "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
        }
        if is_reasoning_model:
            data["reasoning_effort"] = "minimal"
            # Reasoning models use a large chunk of tokens for internal thinking
            # before writing output — so we need a large budget here.
            data["max_completion_tokens"] = self.max_completion_tokens
        else:
            data["temperature"] = self.temperature
            data["top_p"] = self.top_p
            data["max_tokens"] = self.max_tokens


        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to OpenAI ({self.model_name})...")
                response = requests.post(self.url, json=data, headers=headers, timeout=60)

                if response.status_code == 429:
                    try:
                        err_data = response.json()
                        if err_data.get("error", {}).get("type") == "insufficient_quota":
                            msg = err_data.get("error", {}).get("message", "Insufficient Quota")
                            print(f"[!] OpenAI API Error: {msg}")
                            return f"OpenAI Error: {msg}"
                    except Exception:
                        pass
                        
                    wait = float(response.headers.get("Retry-After", 10.0))
                    print(
                        f"[!] OpenAI rate limited. Waiting {wait:.2f}s before retry "
                        f"{attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    try:
                        resp_json = response.json()
                        error_msg = resp_json.get("error", {}).get("message", "Unknown Error")
                        error_type = resp_json.get("error", {}).get("type", "Unknown type")
                        print(f"[!] OpenAI API Error ({response.status_code}) [{error_type}]: {error_msg}")
                        return f"OpenAI Error: {error_msg}"
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                        print(f"[!] OpenAI API Error ({response.status_code}): {error_msg}")
                        return f"OpenAI Error: {error_msg}"

                json_response = response.json()
                choices = json_response.get("choices", [])
                if choices and "message" in choices[0]:
                    msg = choices[0]["message"]
                    content = msg.get("content")
                    refusal = msg.get("refusal")
                    
                    if content:
                        return content
                    if refusal:
                        return f"Model Refusal: {refusal}"

                    finish_reason = choices[0].get("finish_reason", "unknown")
                    usage = json_response.get("usage", {})
                    return f"OpenAI Error: empty response content returned (finish_reason={finish_reason}, usage={usage})"
                
                print(f"[!] OpenAI Error: Unexpected JSON structure: {json_response}")
                return "OpenAI Error: No response content returned."
            except Exception as e:
                print(f"[!] OpenAI Client Exception: {e}")
                return f"OpenAI Error: {e}"

        return "OpenAI Error: Max retries exceeded (rate limit)."


class CloudflareClient(ModelClient):
    """Client for Cloudflare Workers AI (OpenAI-compatible endpoint)."""

    def __init__(self, api_key=None, model_name=DEFAULT_CLOUDFLARE_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS):
        self.api_key = api_key or os.getenv("CLOUDFLARE_API_TOKEN")
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/v1/chat/completions"
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        acct_status = "FOUND" if self.account_id else "NOT_FOUND"
        print(
            f"[*] CloudflareClient initialized (Model: {model_name}, "
            f"Token: {key_status}, Account ID: {acct_status})"
        )

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "Cloudflare Error: API token not found. Please add CLOUDFLARE_API_TOKEN to your .env file."
            print(f"[!] {err}")
            return err

        if not self.account_id:
            err = "Cloudflare Error: Account ID not found. Please add CLOUDFLARE_ACCOUNT_ID to your .env file."
            print(f"[!] {err}")
            return err

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Cloudflare Workers AI ({self.model_name})...")
                response = requests.post(self.url, json=data, headers=headers, timeout=180)

                if response.status_code in {429, 500, 502, 503, 504}:
                    try:
                        error_msg = response.json().get("errors", [{}])[0].get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    if "daily free allocation" in error_msg.lower() or "please upgrade" in error_msg.lower():
                        print(f"[!] Cloudflare quota exhausted: {error_msg}")
                        return f"Cloudflare Error: quota exhausted: {error_msg}"
                    wait = float(response.headers.get("Retry-After", 10.0 + attempt * 10.0))
                    print(
                        f"[!] Cloudflare retryable error ({response.status_code}): {error_msg}. "
                        f"Waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    try:
                        error_msg = response.json().get("errors", [{}])[0].get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    print(f"[!] Cloudflare API Error ({response.status_code}): {error_msg}")
                    return f"Cloudflare Error: {error_msg}"

                json_response = response.json()
                result = json_response.get("result", {})
                choices = result.get("choices") or json_response.get("choices", [])
                if choices and choices[0].get("message"):
                    message = choices[0]["message"]
                    return message.get("content") or message.get("reasoning_content") or ""
                if result.get("response"):
                    return result.get("response", "")
                if result.get("text"):
                    return result.get("text", "")
                return f"Cloudflare Error: No response content returned. Response shape: {json_response}"
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = 10.0 + attempt * 10.0
                print(
                    f"[!] Cloudflare connection error ({type(e).__name__}): {e}. "
                    f"Waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}..."
                )
                time.sleep(wait)
                continue
            except Exception as e:
                print(f"[!] Cloudflare Client Exception: {e}")
                return f"Cloudflare Error: {e}"

        return "Cloudflare Error: Max retries exceeded (rate limit or transient API error)."


class GitHubModelsClient(ModelClient):
    """Client for GitHub Models API (OpenAI-compatible endpoint)."""

    def __init__(self, api_key=None, model_name=DEFAULT_GITHUB_MODEL, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, max_tokens=DEFAULT_MAX_TOKENS):
        self.api_key = api_key or os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_API_KEY")
        self.model_name = model_name
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.url = "https://models.github.ai/inference/chat/completions"
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] GitHubModelsClient initialized (Model: {model_name}, Token: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "GitHub Models Error: token not found. Please add GITHUB_TOKEN to your .env file."
            print(f"[!] {err}")
            return err

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        max_retries = 8
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to GitHub Models ({self.model_name})...")
                response = requests.post(self.url, json=data, headers=headers, timeout=180)

                if response.status_code in {429, 500, 502, 503, 504}:
                    try:
                        error_msg = response.json().get("error", {}).get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(180.0, 15.0 * (attempt + 1))
                    if wait > 1800:
                        print(
                            f"[!] GitHub Models asked us to wait {wait:.1f}s. "
                            "Stopping this request so the benchmark does not hang for hours."
                        )
                        return (
                            "GitHub Models Error: Rate limit retry window is too long "
                            f"({wait:.1f}s). Stop and rerun later or reduce request volume."
                        )
                    print(
                        f"[!] GitHub Models retryable error ({response.status_code}): {error_msg}. "
                        f"Waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    try:
                        error_msg = response.json().get("error", {}).get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    print(f"[!] GitHub Models API Error: {error_msg}")
                    return f"GitHub Models Error: {error_msg}"

                json_response = response.json()
                choices = json_response.get("choices", [])
                if choices and choices[0].get("message"):
                    choice = choices[0]
                    message = choice["message"]
                    content = message.get("content") or message.get("reasoning_content") or ""
                    if str(content).strip():
                        return content
                    finish_reason = choice.get("finish_reason") or "unknown"
                    return (
                        "GitHub Models Error: Response may have been filtered, spent on hidden "
                        "reasoning, or blocked by provider policy. "
                        f"Empty response content returned (finish_reason={finish_reason})."
                    )
                return "GitHub Models Error: No response content returned."
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = min(120.0, 10.0 * (attempt + 1))
                print(
                    f"[!] GitHub Models connection error ({type(e).__name__}): {e}. "
                    f"Waiting {wait:.1f}s before retry {attempt + 1}/{max_retries}..."
                )
                time.sleep(wait)
                continue
            except Exception as e:
                print(f"[!] GitHub Models Client Exception: {e}")
                return f"GitHub Models Error: {e}"

        return "GitHub Models Error: Max retries exceeded (rate limit or transient API error)."


def get_client(provider, **kwargs):
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(model_name=kwargs.get("model_name", DEFAULT_OLLAMA_MODEL))
    elif provider in {"gemini", "google"}:
        return GeminiClient(model_name=kwargs.get("model_name", DEFAULT_GEMINI_MODEL))
    elif provider == "groq":
        return GroqClient(model_name=kwargs.get("model_name", DEFAULT_GROQ_MODEL))
    elif provider == "mistral":
        return MistralClient(model_name=kwargs.get("model_name", DEFAULT_MISTRAL_MODEL))
    elif provider == "openai":
        return OpenAIClient(model_name=kwargs.get("model_name", DEFAULT_OPENAI_MODEL))
    elif provider == "cloudflare":
        return CloudflareClient(model_name=kwargs.get("model_name", DEFAULT_CLOUDFLARE_MODEL))
    elif provider == "github":
        return GitHubModelsClient(model_name=kwargs.get("model_name", DEFAULT_GITHUB_MODEL))
    else:
        raise ValueError(f"Unknown provider: {provider}")

if __name__ == "__main__":
    # Smoke test for Ollama
    client = get_client("ollama")
    print("Testing Generic Client (Ollama)...")
    print(f"Response: {client.generate('Are you online?')}")
