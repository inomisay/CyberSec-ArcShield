import requests
import json
import time
from .models import get_client
from .moderator import apply_moderation

# Default client
_default_client = get_client("ollama")
_moderator = None # Initialize as needed

def send_prompt(prompt, system_prompt=None, use_moderation=False):
    """
    Sends a prompt to the configured AI model client with optional second-layer moderation.
    """
    response = _default_client.generate(prompt, system_prompt)
    
    if use_moderation and _moderator:
        is_safe, moderated_response = apply_moderation(prompt, response, _moderator)
        return moderated_response
        
    return response

if __name__ == "__main__":
    # Test run
    print("Testing connection to Ollama...")
    res = send_prompt("Hello, are you ready for a security test?")
    if res:
        print(f"Success! Response: {res}")
    else:
        print("Failed to connect to Ollama.")
