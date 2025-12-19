import json
import re
from typing import Set, Dict, List
from tqdm import tqdm
from llm_local import LLM_generation


class QwenCoTRechecker:
    def __init__(self):
        pass
    def build_prompt(self, example: Dict) -> str:
        return (
            f"# Question: {example['question']}\n"
            f"# Database: {example['schema']}\n\n"
        )

    def generate_qwen_cot(self, example: Dict) -> str:
        prompt = self.build_prompt(example)
        results = LLM_generation(instruct="You are a Schema Linking Expert\n",prompt=prompt, n=1)
        return results[0] if results else ""


    def parse_key_fields(self, cot_text: str) -> Set[str]:
        key_fields = set()

        # Final declaration
        final_pattern = r"The key field matching the question is:\s*\[(.*?)\]"
        final_match = re.search(final_pattern, cot_text, re.IGNORECASE)
        if final_match:
            for f in final_match.group(1).split(","):
                key_fields.add(f.strip().lower())

        # Step 3
        step3_pattern = r"Key field for filtering:\s*\*\*(.*?)\*\*"
        for match in re.findall(step3_pattern, cot_text, re.IGNORECASE):
            key_fields.add(match.strip().lower())

        return key_fields

    def run(self, input_path: str, output_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        mismatched = []

        for item in tqdm(data, desc="Rechecking with Qwen"):
            example = {
                "question": item["question"],
                "schema": item["schema"]
            }

            original_key_fields = set(
                k.lower() for k in item.get("key_fields", [])
            )

            # Qwen CoT
            qwen_cot = self.generate_qwen_cot(example)
            qwen_key_fields = self.parse_key_fields(qwen_cot)

            # Compare
            if original_key_fields != qwen_key_fields:
                mismatched.append({
                    "question": item["question"],
                    "schema": item["schema"],
                    "original_key_fields": sorted(list(original_key_fields)),
                    "qwen_key_fields": sorted(list(qwen_key_fields)),
                    "original_cot": item["cot"],
                    "qwen_cot": qwen_cot
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mismatched, f, indent=2, ensure_ascii=False)

        print("=" * 60)
        print(f"Total checked: {len(data)}")
        print(f"Mismatched:    {len(mismatched)}")
        print(f"Saved to:      {output_path}")
        print("=" * 60)


def main():
    INPUT_FILE = "data_cot.json"
    OUTPUT_FILE = "data_cot_qwen_mismatch.json"

    checker = QwenCoTRechecker()
    checker.run(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
