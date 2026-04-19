import requests
import json
import os
import re
import time
from huggingface_hub import InferenceClient


def _resolve_gemini_api_key(api_key=None):
    """Resolve Gemini API key from FLASH_LITE or the standard GEMINI_API_KEY."""
    key = os.getenv("GEMINI_FLASH_LITE_API_KEY")
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key

class ModelClient:
    """Base class for AI model clients."""
    def generate(self, prompt, system_prompt=None):
        raise NotImplementedError("Subclasses must implement generate()")

class OllamaClient(ModelClient):
    """Client for local Ollama instances."""
    def __init__(self, model_name="llama3.2", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{base_url}/api/generate"

    def generate(self, prompt, system_prompt=None):
        # Prepend system prompt for simple models or use 'system' field if supported
        final_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        
        data = {
            "model": self.model_name,
            "prompt": final_prompt,
            "stream": False,
            "options": {
                "num_predict": 256,
                "temperature": 0.0
            }
        }
        
        try:
            response = requests.post(self.url, json=data, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Ollama Error: {e}"

class GeminiClient(ModelClient):
    """Client for Google Gemini API."""
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        self.api_key = _resolve_gemini_api_key(api_key)
        self.model_name = model_name
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
        }
        
        # Official system instruction field for Gemini 1.5
        if system_prompt:
            data["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }
        
        max_retries = 4
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Gemini ({self.model_name})...")
                response = requests.post(f"{self.url}?key={self.api_key}", json=data, headers=headers, timeout=60)

                if response.status_code == 429:
                    error_msg = response.json().get('error', {}).get('message', '')
                    match = re.search(r'retry in ([\d.]+)s', error_msg)
                    wait = float(match.group(1)) + 2 if match else 15
                    print(f"[!] Rate limited. Waiting {wait:.1f}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                    continue

                if response.status_code != 200:
                    error_msg = response.json().get('error', {}).get('message', 'Unknown Error')
                    print(f"[!] Gemini API Error: {error_msg}")
                    return f"Gemini Error: {error_msg}"

                json_response = response.json()
                if 'candidates' in json_response and json_response['candidates']:
                    return json_response['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Gemini Error: No response content returned."

            except Exception as e:
                print(f"[!] Gemini Client Exception: {e}")
                return f"Gemini Error: {e}"

        return "Gemini Error: Max retries exceeded (rate limit)."


class GroqClient(ModelClient):
    """Client for Groq API (OpenAI-compatible chat completions)."""

    def __init__(self, api_key=None, model_name="llama-3.1-8b-instant"):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name
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
            "temperature": 0.0,
            # Keep responses compact to reduce TPM pressure during benchmark loops.
            "max_tokens": 128,
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

    def __init__(self, api_key=None, model_name="open-mistral-7b"):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model_name = model_name
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
            "temperature": 0.0,
            "max_tokens": 128,
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
                        error_msg = response.json().get("error", {}).get("message", "Unknown Error")
                    except Exception:
                        error_msg = response.text or "Unknown Error"
                    print(f"[!] Mistral API Error: {error_msg}")
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

    def __init__(self, api_key=None, model_name="gpt-4"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
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
        system_role = "developer" if self.model_name.startswith(("o1", "o3", "gpt-5")) else "system"
        
        if system_prompt:
            messages.append({"role": system_role, "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model_name,
            "messages": messages,
        }
        if self.model_name.startswith(("o1", "o3", "gpt-5")):
            # Reasoning models use a large chunk of tokens for internal thinking
            # before writing output — so we need a large budget here.
            data["max_completion_tokens"] = 4096
        else:
            data["max_tokens"] = 256
            data["temperature"] = 0.0


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
                    
                    return ""
                
                print(f"[!] OpenAI Error: Unexpected JSON structure: {json_response}")
                return "OpenAI Error: No response content returned."
            except Exception as e:
                print(f"[!] OpenAI Client Exception: {e}")
                return f"OpenAI Error: {e}"

        return "OpenAI Error: Max retries exceeded (rate limit)."

class HuggingFaceClient(ModelClient):
    """Client for Hugging Face Inference API using the official SDK."""

    def __init__(self, api_key=None, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.api_key = os.getenv("HUGGING_FACE_API_KEY")
        self.model_name = model_name
        # Point the client specifically to the Hub's serverless OpenAI-compatible path
        # This bypasses the unreliable Provider Router
        self.client = InferenceClient(
            api_key=self.api_key, 
            base_url="https://router.huggingface.co/hf-inference/v1"
        )
        key_status = "FOUND" if self.api_key else "NOT_FOUND"
        print(f"[*] HuggingFaceClient initialized (Model: {model_name}, Key: {key_status})")

    def generate(self, prompt, system_prompt=None):
        if not self.api_key:
            err = "Hugging Face Error: API Key not found. Please add HUGGING_FACE_API_KEY to your .env file."
            print(f"[!] {err}")
            return err

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"[*] Sending request to Hugging Face ({self.model_name})...")
                
                # chat_completion handles the router or direct hub inference automatically
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                
                return response.choices[0].message.content

            except Exception as e:
                # Common errors handle
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    wait = 30.0 + (attempt * 15.0)
                    print(f"[!] HF rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait)
                    continue

                if "503" in err_str or "loading" in err_str.lower():
                    # Hub serverless models might be loading
                    print("[*] HF Model is loading... waiting 25s")
                    time.sleep(25)
                    continue

                # Fallback for Gemma if provider is disabled in router:
                # Try putting :fastest if it's a 400
                if "400" in err_str and "not supported" in err_str and ":" not in self.model_name:
                    print(f"[*] Model {self.model_name} might need a provider suffix. Retrying as {self.model_name}:fastest...")
                    self.model_name = f"{self.model_name}:fastest"
                    continue

                print(f"[!] Hugging Face Client Exception: {e}")
                return f"Hugging Face Error: {e}"

        return "Hugging Face Error: Max retries exceeded (rate limit or loading)."


        return "Hugging Face Error: Max retries exceeded (rate limit or loading)."


def get_client(provider, **kwargs):
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient(model_name=kwargs.get("model_name", "llama3.2"))
    elif provider == "gemini":
        return GeminiClient(model_name=kwargs.get("model_name", "gemini-1.5-flash"))
    elif provider == "groq":
        return GroqClient(model_name=kwargs.get("model_name", "llama-3.1-8b-instant"))
    elif provider == "mistral":
        return MistralClient(model_name=kwargs.get("model_name", "mistral-small-latest"))
    elif provider == "huggingface":
        return HuggingFaceClient(model_name=kwargs.get("model_name", "Qwen/Qwen2.5-7B-Instruct"))
    elif provider == "openai":
        return OpenAIClient(model_name=kwargs.get("model_name", "gpt-4"))
    else:
        raise ValueError(f"Unknown provider: {provider}")

if __name__ == "__main__":
    # Smoke test for Ollama
    client = get_client("ollama")
    print("Testing Generic Client (Ollama)...")
    print(f"Response: {client.generate('Are you online?')}")