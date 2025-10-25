import random
import numpy as np
from datasets import load_dataset
from .hf_auth import resolve_hf_token

def load_gsm8k(split: str, max_samples: int | None = None):
    token = resolve_hf_token()
    ds = load_dataset("openai/gsm8k", "main", token=token)[split]
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
