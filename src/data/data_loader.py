"""Load and normalize the local attack datasets under dataset/ into one combined list.

This loader reads the repository's on-disk dataset folders, applies lightweight
normalization, and returns a unified prompt record structure. It does not fetch
Hugging Face datasets by itself.
"""

import pandas as pd
import json
import os
from pathlib import Path

class DataLoader:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path

    @staticmethod
    def _annotation_metadata(source_name, method, guidelines="docs/CLASSIFICATION_REFERENCE.md", annotator_count="not reported", inter_rater_reliability="not reported"):
        return {
            "annotation_method": method,
            "annotator_count": annotator_count,
            "inter_rater_reliability": inter_rater_reliability,
            "annotation_guidelines": guidelines,
            "ground_truth_provenance": source_name,
        }

    @staticmethod
    def _attack_taxonomy_metadata(row=None, default_vector="unknown", default_mechanism="unknown", default_strategy="unknown"):
        row = row or {}
        return {
            "attack_vector": str(row.get("attack_vector", default_vector)).strip() or default_vector,
            "attack_mechanism": str(row.get("attack_mechanism", default_mechanism)).strip() or default_mechanism,
            "linguistic_strategy": str(row.get("linguistic_strategy", default_strategy)).strip() or default_strategy,
        }

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

    @staticmethod
    def _normalize_bool(value, default=True):
        if isinstance(value, bool):
            return value
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "attack", "malicious", "harmful"}:
            return True
        if text in {"0", "false", "no", "n", "benign", "safe", "harmless"}:
            return False
        return default

    @staticmethod
    def _prompt_from_row(row):
        prompt_keys = [
            "prompt",
            "text",
            "query",
            "question",
            "input",
            "instruction",
            "original_query",
            "persuasive_prompt",
            "variant_query",
            "forbidden_prompt",
        ]
        for key in prompt_keys:
            value = row.get(key)
            if value is None or pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _read_table(self, path):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict) and isinstance(payload.get("records"), list):
                return pd.DataFrame(payload["records"])
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
        return pd.DataFrame()

    def _load_dataset_folder(self, folder_name, source_name, default_vector, exclude_file_names=None):
        folder = Path(self.base_path) / folder_name
        if not folder.exists():
            return []

        attacks = []
        exclude_file_names = {name.lower() for name in (exclude_file_names or [])}
        candidate_files = sorted(folder.rglob("*.csv")) + sorted(folder.rglob("*.json")) + sorted(folder.rglob("*.jsonl"))
        for file_path in candidate_files:
            if file_path.name.lower() in exclude_file_names:
                continue
            try:
                df = self._read_table(file_path)
            except Exception:
                continue

            if df.empty:
                continue

            for _, row in df.iterrows():
                prompt = self._prompt_from_row(row)
                if not prompt:
                    continue

                row_dict = row.to_dict()
                category = self._clean_text(row_dict.get("category") or row_dict.get("intent") or source_name)
                technique = self._clean_text(row_dict.get("technique") or row_dict.get("attack_type") or default_vector)
                is_attack = self._normalize_bool(row_dict.get("is_attack", row_dict.get("label")), default=True)

                attacks.append({
                    "prompt": prompt,
                    "source": source_name,
                    "category": category or source_name,
                    "technique": technique or default_vector,
                    "is_attack": is_attack,
                    **self._attack_taxonomy_metadata(
                        row=row_dict,
                        default_vector=technique or default_vector,
                        default_mechanism="dataset-provided",
                        default_strategy="dataset-provided",
                    ),
                    **self._annotation_metadata(
                        source_name=f"{source_name} ({file_path.name})",
                        method="dataset-provided labels",
                    ),
                })

        return attacks

    def load_educational_guardrails(self):
        return self._load_dataset_folder(
            folder_name="educational-llm-guardrails-bench_Data",
            source_name="Educational Guardrails",
            default_vector="Prompt Injection",
        )

    def load_jailbreak_llms(self):
        return self._load_dataset_folder(
            folder_name="JailbreakLLMs_Data",
            source_name="JailbreakLLMs",
            default_vector="Jailbreak Prompting",
        )

    def load_foot_in_the_door(self):
        return self._load_dataset_folder(
            folder_name="Foot-in-the-door-Jailbreak_Data",
            source_name="Foot-in-the-door",
            default_vector="Role Playing / Pretending",
        )

    def load_largo(self):
        return self._load_dataset_folder(
            folder_name="LARGO_Data",
            source_name="LARGO",
            default_vector="Logic Trap Attacks",
            exclude_file_names={"advbench.csv"},
        )

    def load_magic(self):
        return self._load_dataset_folder(
            folder_name="MAGIC_Data",
            source_name="MAGIC",
            default_vector="Jailbreaks",
        )

    def load_mpa(self):
        return self._load_dataset_folder(
            folder_name="MPA_Data",
            source_name="MPA",
            default_vector="Multi-Prompt Attack",
        )

    def load_strongreject(self):
        return self._load_dataset_folder(
            folder_name="strongreject_Data",
            source_name="StrongREJECT",
            default_vector="Refusal Evasion",
        )

    def load_query_attack(self):
        return self._load_dataset_folder(
            folder_name="QueryAttack_Data",
            source_name="QueryAttack",
            default_vector="Indirect Injection",
        )

    def load_trogen(self):
        return self._load_dataset_folder(
            folder_name="TroGEN_Data",
            source_name="TroGEN",
            default_vector="Obfuscation (Token Smuggling)",
        )

    def get_combined_attacks(
        self,
        include_educational_guardrails=True,
        include_foot_in_the_door=True,
        include_jailbreak_llms=True,
        include_largo=True,
        include_magic=True,
        include_mpa=True,
        include_query_attack=True,
        include_strongreject=True,
        include_trogen=True,
    ):
        # Local-only loader: reads the repository's on-disk dataset folders; Hugging Face ingestion happens elsewhere.
        """Combines attack datasets into a single list."""
        attacks = []
        if include_educational_guardrails:
            attacks.extend(self.load_educational_guardrails())
        if include_foot_in_the_door:
            attacks.extend(self.load_foot_in_the_door())
        if include_jailbreak_llms:
            attacks.extend(self.load_jailbreak_llms())
        if include_largo:
            attacks.extend(self.load_largo())
        if include_magic:
            attacks.extend(self.load_magic())
        if include_mpa:
            attacks.extend(self.load_mpa())
        if include_query_attack:
            attacks.extend(self.load_query_attack())
        if include_strongreject:
            attacks.extend(self.load_strongreject())
        if include_trogen:
            attacks.extend(self.load_trogen())
        return attacks

if __name__ == "__main__":
    loader = DataLoader()
    print(f"Loaded {len(loader.load_educational_guardrails())} Educational Guardrails prompts")
    print(f"Loaded {len(loader.load_foot_in_the_door())} Foot-in-the-door prompts")
    print(f"Loaded {len(loader.load_jailbreak_llms())} JailbreakLLMs prompts")
    print(f"Loaded {len(loader.load_largo())} LARGO prompts")
    print(f"Loaded {len(loader.load_magic())} MAGIC prompts")
    print(f"Loaded {len(loader.load_mpa())} MPA prompts")
    print(f"Loaded {len(loader.load_query_attack())} QueryAttack prompts")
    print(f"Loaded {len(loader.load_strongreject())} StrongREJECT prompts")
    print(f"Loaded {len(loader.load_trogen())} TroGEN prompts")
