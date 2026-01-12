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
            f"# Database: {example['database']}\n\n"
        )
    
    def generate_qwen_cot(self, example: Dict) -> str:
        prompt = self.build_prompt(example)
        results = LLM_generation(
            instruct="You are a Schema Linking Expert\n",
            prompt=prompt, 
            n=1
        )
        return results[0] if results else ""
    
    def parse_think_and_answer(self, cot_text: str) -> tuple:

        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, cot_text, re.DOTALL)
        think = think_match.group(1).strip() if think_match else cot_text
        

        if think_match:
            answer = cot_text[think_match.end():].strip()
        else:
            answer = ""
        
        return think, answer
    
    def parse_schema_links(self, text: str) -> str:

        final_pattern = r"The key field matching the question is:\s*\[(.*?)\]"
        final_match = re.search(final_pattern, text, re.IGNORECASE)
        
        if final_match:
            return final_match.group(1).strip()
        
 
        step3_pattern = r"Key field for filtering:\s*\*\*(.*?)\*\*"
        matches = re.findall(step3_pattern, text, re.IGNORECASE)
        if matches:
            return ", ".join(matches)
        
        return ""
    
    def parse_key_fields(self, cot_text: str) -> Set[str]:

        key_fields = set()
        

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
                "database": item.get("database", item.get("schema", "")) 
            }
            
  
            original_think = item.get("think", item.get("cot", ""))
            original_answer = item.get("answer", "")
            original_schema_links = item.get("schema_links_pred", "")
            
       
            original_key_fields = set()
            if "key_fields" in item:
                original_key_fields = set(k.lower() for k in item["key_fields"])
            else:
             
                original_key_fields = self.parse_key_fields(original_think or original_schema_links)
            
    
            qwen_cot = self.generate_qwen_cot(example)
            qwen_think, qwen_answer = self.parse_think_and_answer(qwen_cot)
            qwen_schema_links = self.parse_schema_links(qwen_cot)
            qwen_key_fields = self.parse_key_fields(qwen_cot)
            

            if original_key_fields != qwen_key_fields:
                mismatched.append({
        
                    "question": item["question"],
                    "database": example["database"],
                    "think_pre": original_think, 
                    "schema_links_pred": original_schema_links, 
                    "answer_pre": original_answer, 
                    
                    "think": qwen_think,  
                    "answer": qwen_answer, 
                    "schema_links": qwen_schema_links, 
                    
                    "original_key_fields": sorted(list(original_key_fields)),
                    "qwen_key_fields": sorted(list(qwen_key_fields)),
                    
                })
        
 
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mismatched, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print(f"Total checked: {len(data)}")
        print(f"Mismatched:    {len(mismatched)}")
        print(f"Match rate:    {(len(data)-len(mismatched))/len(data)*100:.2f}%")
        print(f"Saved to:      {output_path}")
        print("=" * 60)
        
        return mismatched


def main():
    INPUT_FILE = "data_cot.json"
    OUTPUT_FILE = "data_cot_qwen_mismatch.json"
    
    checker = QwenCoTRechecker()
    mismatched_data = checker.run(INPUT_FILE, OUTPUT_FILE)
    
    if mismatched_data:
        print("\nSample mismatched item:")
        print(json.dumps(mismatched_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
