from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict, Any


class ModelInterface:
    def __init__(self, model_name="/path/to/localmodel"):
        """
        Initialize QWen model interface
        
        Args:
            model_name (str): Model path
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model"""
        print(f"Loading model from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded successfully!")

    def LLM_generation(self, instruct: str, prompt: str, n: int = 3,
                       enable_thinking: bool = False, **kwargs) -> List[Dict[str, Any]]:
        """
        Main interface function for generating responses
        
        Args:
            instruct (str): System instruction
            prompt (str): User prompt
            n (int): Number of responses to generate
            enable_thinking (bool): Whether to enable thinking mode
            **kwargs: Additional generation parameters
        
        Returns:
            List[Dict]: List containing generation results
        """
        # Build messages
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Prepare model inputs
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Set default generation parameters
        generation_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 32768),
            "num_beams": kwargs.get("num_beams", 10),
            "num_return_sequences": n,
            "early_stopping": kwargs.get("early_stopping", True),
            "temperature": kwargs.get("temperature", 1.0),
            "length_penalty": kwargs.get("length_penalty", 1.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }
        
        # Generate responses
        generated_ids = self.model.generate(
            **model_inputs,
            **generation_params
        )
        
        # Process generation results
        results = self._process_generated_results(generated_ids, model_inputs, enable_thinking)
        
        return results

    def _process_generated_results(self, generated_ids, model_inputs, enable_thinking: bool) -> List[Dict[str, Any]]:
        """
        Process generated results
        
        Args:
            generated_ids: Generated token ids
            model_inputs: Original inputs
            enable_thinking: Whether thinking mode is enabled
        
        Returns:
            List[Dict]: List of processed results
        """
        input_length = len(model_inputs.input_ids[0])
        results = []
        
        for i, output in enumerate(generated_ids):
            output_ids = output[input_length:].tolist()
            
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            results.append(content)
        
        return results