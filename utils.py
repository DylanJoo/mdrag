import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def update_tokenizer(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-4}GB' # original is -6
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def load_model(model_name_or_path, model_class='causualLM', device='cpu'):

    from transformers import AutoTokenizer
    from models import FiDT5, LlamaForCausalContextLM
    MODEL_CLASS = {"fid": FiDT5, "cepe": LlamaForCausalContextLM}[model_class]
    if model_class == 'cepe':
        model_kwargs = {
            "attn_implementation": "flash_attention_2" if device == 'cuda' else "eager",
            "torch_dtype": torch.bfloat16
        }
    else:
        model_kwargs = {
            "attn_implementation": "sdpa", 
            "torch_dtype": torch.float16
        }

    logger.info(f"Loading {model_name_or_path} ({model_class}) in \
            {model_kwargs['torch_dtype']}...")
    start_time = time.time()

    model = MODEL_CLASS.from_pretrained(
        model_name_or_path, 
        device_map=device,
        **model_kwargs
    )

    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    return model, tokenizer

