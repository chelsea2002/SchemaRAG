from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
import re  # Added for regex pattern matching
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import json


model_name = "/path/to/model/SchemaLinker"
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype='auto',device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)



def extract_key_fields(text):
    # Find the position of the last "The key field"
    last_key = text.rfind("The key field")
    if last_key == -1:
        last_key = text.rfind("The key fields")
        if last_key == -1:
            return 1
    
    # From the last "The key fields", search forward for the first []
    remaining_text = text[last_key:]
    start_bracket = remaining_text.find("[")
    end_bracket = remaining_text.find("]")
    
    if start_bracket == -1 or end_bracket == -1 or end_bracket < start_bracket:
        return 1
    
    # Extract content within []
    content = remaining_text[start_bracket + 1:end_bracket]
    
    # Split content by comma into a list, removing leading/trailing spaces
    result_list = [item.strip() for item in content.split(",") if item.strip()]
    
    return result_list

def extract_think_content(text):
    # Use regex to find content between <think> and </think>
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)  # DOTALL makes . match newlines
    if matches:
        # Return the last occurrence of think content
        print(f"think:{matches[-1].strip()}")
        return matches[-1].strip()
    return ""  # Return empty string if no think content found


def extract_content_after_think(text):
    pattern = r'</think>(.*)'
    match = re.search(pattern, text, re.DOTALL)  # DOTALL makes . match newlines
    print(f"answer:{match.group(1).strip()}")
    return match.group(1).strip() if match else ""  # Return content after </think> or empty string




def llm_talk(text):
    # Convert input text to tensor format
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # Call the generation method to generate new text
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=9192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

with open('/path/to/data/data_with_sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for index, item in enumerate(data):
 
    print(f'generate {index}')

    prompt = f"""
    You are a Schema Linking Expert # Question: {item['question']},\n# Evidence: {item['evidence']},\n# Database: "{item['database']}"
    """


    print(prompt)
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    response = llm_talk(text)
    
    # Extract think content
    think_content = extract_think_content(response)
    # Extract answer content
    answer_content = extract_content_after_think(response)
    schema_links = extract_key_fields(response)
    
    tried = 0
    while schema_links == 1:
        tried += 1
        print(f'we try {tried}')
        response = llm_talk(text)
        print(response)
        # Update think content for each try
        think_content = extract_think_content(response)
        # Extract answer content
        answer_content = extract_content_after_think(response)
        schema_links = extract_key_fields(response)
    
    print(schema_links)
    item['schema_links_pred'] = schema_links
    item['think_pre'] = think_content  # Add think content to the item
    item['answer_pre'] = answer_content
    
    if (index + 1) % 5 == 0 or index == len(data) - 1:
        with open('/path/to/output/data_pre.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)