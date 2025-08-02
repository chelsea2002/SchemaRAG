import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, get_peft_config, ModelConfig
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig


# Local model and dataset paths
model_name = "/path/to/your/qwen_finetune_mtl"
local_dataset_path_train = "/path/to/your/training_data.json"
output_dir = "./SchemaLinker"

class RewardConfig:
    # Adjust reward weights according to requirements
    reward_correct = 2.0        # True Positive (TP): +2
    reward_wrong = -0.5         # False Positive (FP): -0.5  
    reward_missed = -3.0        # False Negative (FN): -3
    f1_bonus_weight = 0.5       # F1 score weight: +0.5
    format_penalty = -1000      # Format error penalty, ensure it's lower than any other reward

reward_config = RewardConfig()

# Reward calculation function
def calculate_format_reward(text):
    """
    Check if the output format is correct.
    Requirements:
    1. Must contain <think></think> tags
    2. Think tags must contain three steps:
       - "1." 
       - "2."
       - "3."
    3. After think tags, must contain "The key fields matching" format
    
    Args:
        text (str): Input text string
        
    Returns:
        bool: True if format is correct, False if format is incorrect
    """
    # Check if contains <think></think> tags
    if "<think>" not in text or "</think>" not in text:
        return False
    
    # Ensure </think> comes after <think>
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    
    if think_end <= think_start:
        return False
    
    # Extract content within think tags
    think_content = text[think_start + 7:think_end]  # 7 is the length of "<think>"
    
    # Check if think content contains required three steps
    required_steps = [
        "1.",
        "2.", 
        "3."
    ]
    
    for step in required_steps:
        if step not in think_content:
            return False
    
    # Check if text after think tags contains "The key fields matching"
    after_think = text[think_end + 8:]  # 8 is the length of "</think>"
    
    if "The key fields matching" not in after_think:
        return False
    
    return True

def calculate_step_reward(pred_links, true_links, config):
    """
    Calculate reward score based on prediction accuracy
    
    Args:
        pred_links (set): Set of predicted links
        true_links (set): Set of true links
        config: Reward configuration object
    
    Returns:
        tuple: (reward score, detailed information dictionary)
    """
    # Calculate various prediction results
    true_positive = len(pred_links & true_links)    # True Positive: number of correct predictions
    false_positive = len(pred_links - true_links)   # False Positive: number of incorrect predictions  
    false_negative = len(true_links - pred_links)   # False Negative: number of missed predictions
    
    # Calculate precision and recall
    precision = true_positive / max(1, len(pred_links)) if pred_links else 0
    recall = true_positive / max(1, len(true_links)) if true_links else 0
    f1_score = 2 * precision * recall / max(1e-8, precision + recall)
    
    # Calculate total reward based on new reward strategy
    reward = (
        config.reward_correct * true_positive +     # True positive reward: +2 * correct count
        config.reward_wrong * false_positive +      # False positive penalty: -0.5 * incorrect count
        config.reward_missed * false_negative +     # False negative heavy penalty: -3 * missed count
        config.f1_bonus_weight * f1_score          # F1 score bonus: +0.5 * F1 score
    )
    
    info = {
        "true_positive": true_positive,
        "false_positive": false_positive, 
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "base_reward": reward
    }
    return reward, info

# GRPO reward function
def grpo_reward_func(completions, **kwargs):
    """
    Reward function for GRPO training
    First check format, only calculate content reward if format is correct, otherwise give minimum reward
    """
    rewards = []
    true_links = kwargs['true_links']
    
    for i in range(len(completions)):
        gen_text = completions[i]
        true_link = true_links[i]
        
        print(f"Generated text: {gen_text}")
        
        # First check if format is correct
        is_format_correct = calculate_format_reward(gen_text)
        
        if not is_format_correct:
            # Format incorrect, give minimum reward, cannot get any subsequent rewards
            reward = reward_config.format_penalty
            print(f"Format incorrect! Missing required structure:")
            print(f"  - Must have <think></think> tags")
            print(f"  - Must contain '1. Understand', '2. Analyze', '3. Determine' in <think>")
            print(f"  - Must have 'The key fields matching' after </think>")
            print(f"Penalty reward: {reward}")
        else:
            # Format correct, continue calculating content reward
            pred_links = extract_key_fields(gen_text)
            print(f"Predicted links: {pred_links}")
            print(f"True links: {true_link}")
            
            content_reward, reward_info = calculate_step_reward(pred_links, true_link, reward_config)
            reward = content_reward
            
            print(f"Reward breakdown:")
            print(f"  True Positive: {reward_info['true_positive']} × {reward_config.reward_correct} = {reward_info['true_positive'] * reward_config.reward_correct}")
            print(f"  False Positive: {reward_info['false_positive']} × {reward_config.reward_wrong} = {reward_info['false_positive'] * reward_config.reward_wrong}")
            print(f"  False Negative: {reward_info['false_negative']} × {reward_config.reward_missed} = {reward_info['false_negative'] * reward_config.reward_missed}")
            print(f"  F1 Bonus: {reward_info['f1_score']:.3f} × {reward_config.f1_bonus_weight} = {reward_info['f1_score'] * reward_config.f1_bonus_weight:.3f}")
            print(f"  Total content reward: {reward}")
        
        rewards.append(reward)
        print(f"Final reward for completion {i}: {reward}")
        print("-" * 50)
    
    return rewards

# Extract key fields
def extract_key_fields(text):
    """
    Extract key fields from generated text
    Find the last occurrence of "The key field" or "The key fields", then extract content within brackets
    """
    last_key = text.rfind("The key field")
    if last_key == -1:
        last_key = text.rfind("The key fields")
        if last_key == -1:
            return set()  # Return empty set
    
    remaining_text = text[last_key:]
    start_bracket = remaining_text.find("[")
    end_bracket = remaining_text.find("]")
    
    if start_bracket == -1 or end_bracket == -1 or end_bracket < start_bracket:
        return set()
    
    content = remaining_text[start_bracket + 1:end_bracket]
    result_list = {item.strip() for item in content.split(",") if item.strip()}
    return result_list

class LocalJSONDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        for item in datas:
            processed_data = self.process_func(item)
            self.data.append(processed_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def process_func(self, example):
        prompt_text = (
            f"<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n"
            f"<|im_start|>user\n # Question:{example['question']}\n #Evidence:{example['evidence']} \n # Database:{example['database']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return {
            "prompt": prompt_text,  # Return string, not token IDs
            "true_links": set(example['schema_links'])
        }

# Configuration parameters
grpo_config = GRPOConfig(
    learning_rate=1.41e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,  
    output_dir=output_dir,
    logging_steps=10
    )


def main(model_args):

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    # Create dataset
    train_dataset = LocalJSONDataset(
        local_dataset_path_train,
        tokenizer,
        max_length=512
    )

    # Data collation function
    def collate_fn(batch):
        prompts = [item["prompt"] for item in batch]  # Directly extract string list
        true_links = [item["true_links"] for item in batch]
        return {
            "prompt": prompts,  # Return List[str]
            "true_links": true_links
        }

    # Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,
        args=grpo_config,
        train_dataset=train_dataset, 
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args)
    )
    # Training
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    # Save model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    model_args = ModelConfig(
        use_peft=True,
        lora_r=64,
        lora_alpha=32,
        lora_dropout=0.1
    )
    main(model_args)