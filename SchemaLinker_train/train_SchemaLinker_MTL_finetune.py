import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import numpy as np
import random

class LocalJSONDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        """
        Load multi-task dataset from data list

        Args:
            data_list (list): List of data items
            tokenizer: Tokenizer
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Ensure tokenizer has correct special tokens
        special_tokens = {
            "pad_token": "<|padding|>",
            "eos_token": "<|endoftext|>",
        }
        self.tokenizer.add_special_tokens(
            {"pad_token": special_tokens["pad_token"]} if self.tokenizer.pad_token is None else {}
        )
        
        # Encode all samples
        for item in data_list:
            processed_data = self.process_func(item)
            self.data.extend(processed_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def process_func(self, example):
        # System prompt
        system_prompt = "<|im_start|>system\nYou are a Schema Linking Expert<|im_end|>\n"
        
        # Task mapping for adding task identifiers during preprocessing
        task_mapping = {
            "error_detection": 0,
            "correction": 1,
            "generation": 2,
        }
        
        # Define input and target text for each task
        tasks = {
            "error_detection": (
                f"{system_prompt}<|im_start|>user\n # Question:{example['question']}\n # Database:{example['database']}\n"
                f"# Wrong Reasoning:{example['think_pre']}\n# Schema Links:{example['schema_links_pred']}\nPlease analyze the error:<|im_end|>\n<|im_start|>assistant\n",
                f"{example['error_explanation']}<|im_end|>"
            ),
            "correction": (
                f"{system_prompt}<|im_start|>user\n # Question:{example['question']}\n # Database:{example['database']}\n"
                f"# Wrong Reasoning:{example['think_pre']}\n# Wrong Answer:{example['schema_links_pred']}\n# Error Analysis:{example['error_explanation']}\n"
                f"Please provide the correct reasoning and answer:<|im_end|>\n<|im_start|>assistant\n",
                f"<think>{example['think']}</think>{example['answer']}<|im_end|>"
            ),
            "generation": (
                f"{system_prompt}<|im_start|>user\n # Question:{example['question']}\n # Database:{example['database']}\n"
                f"<|im_end|>\n<|im_start|>assistant\n",
                f"<think>{example['think']}</think>{example['answer']}<|im_end|>"
            )
        }

        processed_data = []
        
        for task_name, (input_text, target_text) in tasks.items():
            # Add task identifier prefix to input text
            task_id = task_mapping[task_name]
            task_prefix = f"<task:{task_id}>"  # Create task-specific prefix
            prefixed_input = task_prefix + input_text
            
            # Encode input text
            input_encoding = self.tokenizer(prefixed_input, add_special_tokens=False)
            target_encoding = self.tokenizer(target_text, add_special_tokens=False)
            
            # Build complete sequence input_ids and attention_mask
            input_ids = input_encoding["input_ids"]
            attention_mask = input_encoding["attention_mask"]
            
            # Build labels: input part as -100, output part as target token ids
            # Task prefix part is also set to -100 to not calculate loss
            labels = [-100] * len(input_ids)
            
            # Concatenate target text
            target_ids = target_encoding["input_ids"]
            target_mask = target_encoding["attention_mask"]
            
            # Concatenate input and target
            input_ids = input_ids + target_ids
            attention_mask = attention_mask + target_mask
            labels = labels + target_ids
            
            # Ensure not exceeding maximum length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                labels = labels[:self.max_length]
            else:
                # Pad to maximum length
                padding_length = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * padding_length
                attention_mask += [0] * padding_length
                labels += [-100] * padding_length
            
            processed_data.append({
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
                "task_id": task_id
            })

        return processed_data


class CustomTrainer(Trainer):
    """
    Custom trainer for full fine-tuning with task-balanced sampling.
    
    This trainer implements task-balanced sampling to ensure equal representation
    of different tasks during training. Unlike parameter-efficient methods,
    this trainer performs full fine-tuning of all model parameters.
    """
    
    def __init__(self, *args, **kwargs):
        # Save task weight configuration for logging but not directly used for loss calculation
        self.w_error = kwargs.pop("w_error", 0.3)
        self.w_correction = kwargs.pop("w_correction", 0.3) 
        self.w_generation = kwargs.pop("w_generation", 1.0)
        
        super().__init__(*args, **kwargs)
    
    # No longer override compute_loss method, use Hugging Face default loss calculation

    def _get_train_sampler(self):
        """Implement task-balanced sampling strategy"""
        if self.train_dataset is None or not hasattr(self.train_dataset, "data"):
            return super()._get_train_sampler()
            
        # Analyze task distribution in dataset
        task_counts = {}
        for item in self.train_dataset.data:
            task_id = item.get("task_id", 0)
            task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
        # Calculate task weights (inverse of task frequency)
        task_weights = {}
        total_samples = len(self.train_dataset)
        for task_id, count in task_counts.items():
            task_weights[task_id] = total_samples / (len(task_counts) * count)
            
        # Set additional weight multipliers for different task types
        task_multipliers = {
            0: self.w_error,         # error_detection task
            1: self.w_correction,    # correction task
            2: self.w_generation,    # generation task
        }
        
        # Create sample weight list, applying task type multipliers
        sample_weights = []
        for item in self.train_dataset.data:
            task_id = item.get("task_id", 0)
            base_weight = task_weights[task_id]
            multiplier = task_multipliers.get(task_id, 1.0)
            sample_weights.append(base_weight * multiplier)
        
        # Use WeightedRandomSampler
        if self.args.world_size <= 1:
            return WeightedRandomSampler(
                sample_weights,
                len(sample_weights),
                replacement=True
            )
        else:
            # Use DistributedSampler for distributed training
            return super()._get_train_sampler()


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
    local_model_path = "/path/to/your/qwen_finetune_CoT"
    local_dataset_path_train = "/path/to/your/data/train.json"
    output_dir = "./qwen_finetune_mtl"
    
    # Device configuration and memory optimization
    torch.cuda.empty_cache()
    
    # For full fine-tuning, you may want to enable these optimizations:
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Load local tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    # Ensure tokenizer has necessary special tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    
    # Load and configure model for full fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize token embeddings if tokenizer has new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Enable gradient computation for all parameters (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True
    
    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
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
    train_dataset = LocalJSONDataset(train_data, tokenizer, max_length=512)
    val_dataset = LocalJSONDataset(val_data, tokenizer, max_length=512)
    
    # Count task distribution in training set
    task_counts = {"error_detection": 0, "correction": 0, "generation": 0, "wrong": 0}
    for sample in train_dataset:
        task_id = sample.get("task_id", 0)
        task_name = {0: "error_detection", 1: "correction", 2: "generation", 3: "wrong"}[task_id]
        task_counts[task_name] += 1
    
    print("Task distribution in training set:", task_counts)
    
    # Define task weights - adjust based on task importance and data distribution
    w_error = 0.3
    w_correction = 0.3  # Slightly increase weight for error correction task
    w_generation = 1.0  # Maintain high weight for generation task
    
    # Training arguments optimized for full fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Reduced epochs for full fine-tuning
        per_device_train_batch_size=2,  # Smaller batch size due to memory constraints
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        learning_rate=5e-6,  # Much lower learning rate for full fine-tuning
        weight_decay=0.01,
        max_grad_norm=1.0,  # Gradient clipping
        fp16=False,
        bf16=True,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",  # More frequent evaluation
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,  # Save every 100 steps
        save_total_limit=3,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,  # Specify smaller eval_loss is better
        label_names=["labels"],
        # Add warmup steps
        warmup_ratio=0.05,  # Reduced warmup for full fine-tuning
        # Add learning rate scheduling
        lr_scheduler_type="cosine",
        # Memory optimization
        dataloader_pin_memory=False,
        # Enable gradient checkpointing to save memory
        gradient_checkpointing=True,
        # Remove unused columns to save memory
        remove_unused_columns=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize custom Trainer for full fine-tuning
    # Note: Full fine-tuning requires significantly more GPU memory than LoRA
    # Consider using DeepSpeed ZeRO for very large models or multiple GPUs
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # Pass weight parameters to loss function
        w_error=w_error,
        w_correction=w_correction,
        w_generation=w_generation,
    )
    
    # Start training
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training completed! Model saved to: {output_dir}")
    
    # Model evaluation
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Simple inference test on validation set
    test_sample = val_dataset[0]
    input_ids = test_sample["input_ids"].unsqueeze(0).cuda()
    attention_mask = test_sample["attention_mask"].unsqueeze(0).cuda()
    
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        # Generate output using greedy decoding
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )
    
    # Decode generated content
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    print("\nTest generation result example:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()