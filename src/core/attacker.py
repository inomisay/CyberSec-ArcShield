import time
from datetime import UTC, datetime

from .models import get_client
from .moderator import apply_moderation

# Default client
_default_client = get_client("ollama")
_moderator = None # Initialize as needed


def parse_model_spec(model_spec):
    """Split provider-prefixed model specs such as `google:gemini-flash-lite-latest`."""
    if ":" not in str(model_spec):
        return "ollama", str(model_spec)
    provider, model_name = str(model_spec).split(":", 1)
    return provider.strip(), model_name.strip()


def canonical_output_model_spec(model_spec):
    """Keep Gemma output identity stable when Cloudflare credential profiles change."""
    provider, model_name = parse_model_spec(model_spec)
    cloudflare_aliases = {"cloudflare", "cloudflare2", "cloudflare_alt", "cloudflare_gemma"}
    if provider.lower() in cloudflare_aliases and model_name == "@cf/google/gemma-4-26b-a4b-it":
        return f"cloudflare:{model_name}"
    return str(model_spec)


def configure_client(client, temperature=None, top_p=None, max_tokens=None):
    """Apply deterministic generation settings where the client supports them."""
    if temperature is not None and hasattr(client, "temperature"):
        client.temperature = temperature
    if top_p is not None and hasattr(client, "top_p"):
        client.top_p = top_p
    if max_tokens is not None and hasattr(client, "max_tokens"):
        client.max_tokens = max_tokens
    if max_tokens is not None and hasattr(client, "max_completion_tokens"):
        client.max_completion_tokens = max_tokens
    return client


def get_attack_client(
    model_spec="ollama:llama3.1:8b",
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
    cloudflare_credentials="model",
):
    """Create a model client for benchmark execution."""
    provider, model_name = parse_model_spec(model_spec)
    client = get_client(
        provider,
        model_name=model_name,
        cloudflare_credentials=cloudflare_credentials,
    )
    return configure_client(client, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def approximate_token_count(text):
    """Lightweight token proxy used for reproducible benchmark logging."""
    return len(str(text or "").split())


def send_prompt(prompt, system_prompt=None, use_moderation=False, client=None):
    """
    Sends a prompt to the configured AI model client with optional second-layer moderation.
    """
    active_client = client or _default_client
    response = active_client.generate(prompt, system_prompt)
    
    if use_moderation and _moderator:
        is_safe, moderated_response = apply_moderation(prompt, response, _moderator)
        return moderated_response
        
    return response


def execute_attack(
    prompt,
    model_spec="ollama:llama3.1:8b",
    system_prompt=None,
    client=None,
    temperature=0.0,
    top_p=1.0,
    max_tokens=512,
    use_moderation=False,
):
    """Run one benchmark prompt and return response metadata for the judge."""
    provider, model_name = parse_model_spec(model_spec)
    active_client = client or get_attack_client(model_spec, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    started = time.perf_counter()
    timestamp = datetime.now(UTC).isoformat()
    error = ""
    try:
        response = send_prompt(prompt, system_prompt=system_prompt, use_moderation=use_moderation, client=active_client)
    except Exception as exc:
        error = str(exc)
        response = f"Provider Error: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    prompt_tokens = approximate_token_count(prompt)
    context_tokens = prompt_tokens + approximate_token_count(system_prompt or "")
    completion_tokens = approximate_token_count(response)

    return {
        "provider": provider,
        "model_name": model_name,
        "model_spec": model_spec,
        "model_response": str(response),
        "error": error,
        "inference_timestamp": timestamp,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "context_tokens": context_tokens,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "context_overhead": context_tokens - prompt_tokens,
        "token_overhead": completion_tokens,
    }

if __name__ == "__main__":
    # Test run
    print("Testing connection to Ollama...")
    res = send_prompt("Hello, are you ready for a security test?")
    if res:
        print(f"Success! Response: {res}")
    else:
        print("Failed to connect to Ollama.")
