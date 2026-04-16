import os
import random
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import DataLoader

def export_prompts(num_per_source=40):
    loader = DataLoader(base_path="data")
    
    # 1. Load all sources
    sources = {
        "Malignant": loader.load_malignant(),
        "Chatbot Safety": loader.load_chatbot_safety(),
        "Prompt Engineering": loader.load_prompt_engineering_attacks(),
        "Fruit Injection Analysis": loader.load_fruit_injection_analysis(),
    }
    
    output_path = Path("logs/manual_prompts_to_test.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Exporting {num_per_source} random prompts from each source...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== MANUAL RED-TEAM PROMPTS FOR CHATGPT TESTING ===\n")
        f.write(f"Generated on: {os.popen('date /t').read().strip()}\n")
        f.write("Instructions: Copy a prompt, paste it into ChatGPT, then use your Extension to log the result.\n\n")
        
        for source_name, prompts in sources.items():
            f.write(f"\n--- SOURCE: {source_name} ---\n")
            
            if not prompts:
                f.write("[!] No prompts found for this source.\n")
                continue
                
            # Sample random prompts
            sample_size = min(num_per_source, len(prompts))
            sampled = random.sample(prompts, sample_size)
            
            for i, p_data in enumerate(sampled, 1):
                prompt_text = p_data.get("prompt", "").replace("\n", " ").strip()
                f.write(f"{i}. {prompt_text}\n")
                f.write("-" * 20 + "\n")
                
    print(f"[+] Successfully exported {sum(min(num_per_source, len(p)) for p in sources.values())} prompts to: {output_path}")

if __name__ == "__main__":
    export_prompts()
