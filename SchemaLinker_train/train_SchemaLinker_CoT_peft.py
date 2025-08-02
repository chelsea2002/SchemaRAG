import os
import torch
import re
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from torch.utils.data import Dataset
import json
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from transformers import EarlyStoppingCallback
import random

import json

class LocalJSONDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        """
        Load dataset from data list

        Args:
            data_list (list): List of data items
            tokenizer: Tokenizer
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Encode all texts
        for item in data_list:
            processed_data = self.process_func(item)
            self.data.append(processed_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def process_func(self, example):
        input_ids, attention_mask, labels = [], [], []
        
        # Encode instruction and response
        instruction = self.tokenizer(f"<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n<|im_start|>user\n # Question:{example['question']}\n # Database:{example['database']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        response = self.tokenizer(f"<think>{example['think']}</think>{example['answer']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # EOS token should also be attended to
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        
        # Truncate and pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        else:
            # Padding to max_length
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([self.tokenizer.pad_token_id] * padding_length)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def split_dataset(data_list, train_ratio=0.9, random_seed=42):
    """
    Split dataset into train and validation sets
    
    Args:
        data_list (list): List of data items
        train_ratio (float): Ratio for training set (default: 0.9)
        random_seed (int): Random seed for reproducible splits
    
    Returns:
        tuple: (train_data, val_data)
    """
    # Set random seed for reproducible results
    random.seed(random_seed)
    
    # Shuffle the data
    shuffled_data = data_list.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split index
    split_idx = int(len(shuffled_data) * train_ratio)
    
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    return train_data, val_data


def main():
    # Local model and dataset paths
    local_model_path = "/path/to/your/model"  # Local model folder path
    local_dataset_path_train = "/path/to/your/data/train_data.json"  # Local training data JSON file path
    output_dir = "./qwen_finetune_CoT"

    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True  # Alternative to `load_in_8bit=True`
    # )

    # Load local tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=64,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Load and split dataset
    print("Loading dataset...")
    with open(local_dataset_path_train, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    print(f"Total data samples: {len(all_data)}")
    
    # Split dataset into train and validation (9:1 ratio)
    train_data, val_data = split_dataset(all_data, train_ratio=0.9, random_seed=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = LocalJSONDataset(
        train_data, 
        tokenizer,
        max_length=512
    )

    val_dataset = LocalJSONDataset(
        val_data,
        tokenizer, 
        max_length=512   
    )   


    # Set training epochs - this is the maximum number of epochs, early stopping may halt training before reaching this number
    num_epochs = 10  # Set a larger maximum number of epochs

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save model at the end of each epoch
        save_total_limit=3,     # Keep the latest 3 checkpoints
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        label_names=["labels"]
    )


    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Initialize Trainer with early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 epochs
    )
    
    # Start training
    trainer.train()
    
    # Save fine-tuned model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning completed! Model saved to:", output_dir)

if __name__ == "__main__":
    main()