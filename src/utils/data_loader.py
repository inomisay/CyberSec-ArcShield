import pandas as pd
import json
import os

class DataLoader:
    def __init__(self, base_path="data"):
        self.base_path = base_path

    def _first_existing_path(self, *parts):
        """Resolve dataset paths from extracted folders, with optional Kaggel fallback."""
        candidates = [
            os.path.join(self.base_path, *parts),
            os.path.join(self.base_path, "Kaggel", *parts),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _clean_text(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    def load_malignant(self):
        """Loads the Malignant dataset from CSV."""
        path = self._first_existing_path("Prompt Injection Malignant", "malignant.csv")
        if not os.path.exists(path):
            return []
        
        df = pd.read_csv(path)
        # Use 'text' as the prompt
        return [
            {
                "prompt": row['text'],
                "source": "Malignant",
                "category": row['category'],
                "technique": "Direct Injection"
            }
            for _, row in df.iterrows()
        ]

    def load_chatbot_safety(self):
        """Loads the Chatbot Safety adversarial dataset."""
        path = self._first_existing_path(
            "LLM jailbreak + safety data for chatbot apps",
            "adversarial_dataset_with_techniques.csv",
        )
        if not os.path.exists(path):
            return []
            
        df = pd.read_csv(path)

        # Columns: original_query,variant_query,persuasive_prompt,technique,intent
        attacks = []
        for _, row in df.iterrows():
            # Use 'persuasive_prompt' as primary, fallback to 'variant_query' then 'original_query'
            prompt = self._clean_text(row.get('persuasive_prompt'))
            if not prompt:
                prompt = self._clean_text(row.get('variant_query'))
            if not prompt:
                prompt = self._clean_text(row.get('original_query'))

            if not prompt:
                continue
            
            attacks.append({
                "prompt": prompt,
                "source": "Chatbot Safety",
                "category": row.get('intent', 'Adversarial'),
                "technique": row.get('technique', 'Adversarial')
            })
        return attacks

    def load_prompt_engineering_attacks(self):
        """Loads prompt examples as benchmark prompts from Prompt Engineering dataset."""
        primary_path = self._first_existing_path(
            "Prompt Engineering Dataset",
            "prompt_examples_dataset.csv",
        )
        fallback_path = self._first_existing_path(
            "Prompt Engineering Dataset",
            "Prompt_Examples.csv",
        )

        attacks = []

        if os.path.exists(primary_path):
            df = pd.read_csv(primary_path)
            for _, row in df.iterrows():
                prompt = self._clean_text(row.get("bad_prompt"))
                if not prompt:
                    prompt = self._clean_text(row.get("task_description"))
                if not prompt:
                    continue

                attacks.append({
                    "prompt": prompt,
                    "source": "Prompt Engineering",
                    "category": self._clean_text(row.get("complexity")) or "Prompt Design",
                    "technique": self._clean_text(row.get("prompting_techniques")) or "Prompt Engineering",
                })

        if attacks:
            return attacks

        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            for _, row in df.iterrows():
                prompt = self._clean_text(row.get("original_prompt"))
                if not prompt:
                    continue

                attacks.append({
                    "prompt": prompt,
                    "source": "Prompt Engineering",
                    "category": "Prompt Design",
                    "technique": "Prompt Engineering",
                })

        return attacks

    def load_fruit_injection_analysis(self):
        """Loads fruit-injection analysis records and converts them into benchmark prompts."""
        path = self._first_existing_path(
            "LLM Jailbreaks Prompt Fruit Injection Analysis",
            "christine_classy_legacy_protected.json",
        )
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        records = payload.get("records", [])
        attacks = []

        for record in records:
            platform = self._clean_text(record.get("platform")) or "Unknown Platform"
            scene = self._clean_text(record.get("scene")) or "Unknown Scene"
            date = self._clean_text(record.get("date")) or "Unknown Date"
            recognition = self._clean_text(record.get("recognition")) or "Unknown Recognition"

            prompt = (
                "You are analyzing a prompt injection incident summary. "
                f"Platform: {platform}. Scene: {scene}. Date: {date}. Recognition: {recognition}. "
                "Explain possible attack vectors and provide a mitigation checklist."
            )

            attacks.append({
                "prompt": prompt,
                "source": "Fruit Injection Analysis",
                "category": recognition,
                "technique": "Prompt Fruit Injection",
            })

        return attacks

    def load_prompt_engineering_examples(self):
        """Loads examples of weak vs strong prompts for defense refinement."""
        path = self._first_existing_path("Prompt Engineering Dataset", "Prompt_Examples.csv")
        if not os.path.exists(path):
            return []
            
        df = pd.read_csv(path)
        # Columns: original_prompt,context,instruction,has_context,conversation_id,Base_Prompt,V1_Prompt,V2_instruction
        return [
            {
                "original_prompt": row.get('original_prompt', ''),
                "refined_prompt": row.get('Base_Prompt', ''),
                "context": row.get('context', ''),
                "instruction": row.get('instruction', '')
            }
            for _, row in df.iterrows()
        ]

    def get_combined_attacks(
        self,
        include_malignant=True,
        include_chatbot_safety=True,
        include_prompt_engineering=True,
        include_fruit_injection=True,
    ):
        """Combines attack datasets into a single list."""
        attacks = []
        if include_malignant:
            attacks.extend(self.load_malignant())
        if include_chatbot_safety:
            attacks.extend(self.load_chatbot_safety())
        if include_prompt_engineering:
            attacks.extend(self.load_prompt_engineering_attacks())
        if include_fruit_injection:
            attacks.extend(self.load_fruit_injection_analysis())
        return attacks

if __name__ == "__main__":
    loader = DataLoader()
    print(f"Loaded {len(loader.load_malignant())} Malignant prompts")
    print(f"Loaded {len(loader.load_chatbot_safety())} Chatbot Safety prompts")
    print(f"Loaded {len(loader.load_prompt_engineering_attacks())} Prompt Engineering prompts")
    print(f"Loaded {len(loader.load_fruit_injection_analysis())} Fruit Injection Analysis prompts")
    print(f"Loaded {len(loader.load_prompt_engineering_examples())} Prompt Engineering examples")
