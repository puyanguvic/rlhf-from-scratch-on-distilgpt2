"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    return AutoTokenizer.from_pretrained(model_name)

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM

def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    return torch.argmax(logits).item()

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    
    scaled_logits = logits / temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    token_id = torch.multinomial(probabilities, num_samples=1)

    return token_id.item()

