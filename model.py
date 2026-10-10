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

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    return [
        [1 if token_id != pad_id else 0 for token_id in sequence]
        for sequence in padded_ids
    ]

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    input_sequences = [example["input_ids"] for example in batch]
    label_sequences = [example["labels"] for example in batch]

    padded_ids = pad_batch(input_sequences, pad_id)
    padded_labels = pad_batch(label_sequences, -100)
    attention_mask = make_attention_mask(padded_ids, pad_id)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

# Step 18 - iterate_minibatches
import random

def iterate_minibatches(examples, batch_size, seed=0):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start:start + batch_size]

# Step 19 - train_val_split
import random

def train_val_split(examples, val_ratio=0.2, seed=0):
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)

    val = shuffled[:val_size]
    train = shuffled[val_size:]

    return train, val

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    shift_logits = logits[:,:-1, :]
    shift_labels = labels[:, 1:]

    return shift_logits, shift_labels

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    vocab_size = shift_logits.shape[-1]

    return F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    beta1, beta2 = betas

    with torch.no_grad():
        if "step" not in state:
            state["step"] = 0
            state["m"] = torch.zeros_like(param)
            state["v"] = torch.zeros_like(param)

        state["step"] += 1
        step = state["step"]

        state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
        state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        m_hat = state["m"] / (1 - beta1 ** step)
        v_hat = state["v"] / (1 - beta2 ** step)

        param.mul_(1 - lr * weight_decay)

        param.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
    
    return param

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    
    return max(0.0, min(1.0, step / warmup_steps))

# Step 24 - clip_grad_norm
import torch

def clip_grad_norm(grads, max_norm):
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")
    
    grad = [g for g in grads if g is not None]

    total_norm = sum(
        g.detach().double().square().sum().item()
        for g in grads
    ) ** 0.5

    if total_norm > max_norm:
        scale = max_norm / total_norm

        with torch.no_grad():
            for g in grads:
                g.mul_(scale)
    
    return total_norm

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    return torch.stack(grad_list, dim=0).mean(dim=0)

