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

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    if k < 1:
        raise ValueError("k must be poe")
    
    k = min(k, logits.shape[-1])
    
    values, indices = torch.topk(logits, k, dim=-1)

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=indices, src=values)

    return filtered_logits

# Step 8 - top_p_filter
def top_p_filter(logits, p):

    logits = torch.as_tensor(logits)
    if not logits.is_floating_point():
        logits = logits.float()
    
    if not 0 <=p <= 1:
        raise ValueError("p must be between 0 and 1")

    if p == 1:
        return logits.clone()
    
    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1
    )

    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probabilities, dim=-1)

    remove_mask = cumulative_probs >= p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove_mask, float("-inf")
    )

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(
        dim=-1, index=sorted_indices, src=sorted_logits
    )

    return filtered_logits

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    return [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris.",
        },
        {
            "prompt": "Calculate 2 + 3.",
            "response": "2 + 3 = 5.",
        },
        {
            "prompt": "Translate 'hello' into French.",
            "response": "Bonjour.",
        },
        {
            "prompt": "Explain what a tokenizer does.",
            "response": "A tokenizer converts text into tokens and their numerical IDs.",
        },
        {
            "prompt": "Write a Python function that adds two numbers.",
            "response": "def add(a, b):\n    return a + b",
        },
    ]

# Step 10 - format_example
def format_example(example):
    return (
        f"### Instruction:\n{example['prompt']}\n\n"
        f"### Response:\n{example['response']}"
    )

# Step 11 - apply_template
def apply_template(examples):
    return [format_example(example) for example in examples]

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    return tokenizer.encode(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

# Step 13 - build_labels
def build_labels(input_ids):
    return input_ids.copy()

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    masked_labels = labels.copy()

    for i in range(min(prompt_length, len(masked_labels))):
        masked_labels[i] = -100
    
    return masked_labels

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    max_length = max((len(seq) for seq in sequences), default=0)

    return [
        seq + [pad_id] * (max_length - len(seq))
        for seq in sequences
    ]

