import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[!] No Gemini API Key found in .env!")
    exit()

print(f"[*] Checking models for key: {api_key[:8]}...")

# Try checking both v1 and v1beta
for version in ["v1", "v1beta"]:
    print(f"\n--- Checking Version: {version} ---")
    url = f"https://generativelanguage.googleapis.com/{version}/models?key={api_key}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not models:
                print(f"[?] No models returned for {version}")
            for m in models:
                # We only care about models that support generateContent
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    print(f"  > {m['name']}")
        else:
            print(f"[!] Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[!!] Request failed: {e}")

print("\n[*] Which of these names would you like to use for the benchmark?")
