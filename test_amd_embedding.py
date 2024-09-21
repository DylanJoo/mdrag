import torch
from datasets import load_dataset
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

dataset = load_dataset('BeIR/scidocs', 'corpus', keep_in_memory=True)['corpus']
documents = dataset['text'][:64]
prompts = [f"Document:{d}\nDocument again:{d}" for d in documents]

model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

with torch.device("cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        trust_remote_code=True
    )
with torch.no_grad():

    print("Start")
    start_ = time.time()
    for i, prompt in enumerate(prompts):
        if i == 0:
            print(prompt)

        batch = tokenizer.batch_encode_plus(
            [prompt],
            return_tensors='pt',
            padding=True,
            max_length=1024,
            truncation=True
        ).to(model.device)
        n = batch['input_ids'].shape[-1]

        start = time.time()
        hidden_states = model(
            output_hidden_states=True,
            return_dict=True,
            **batch
        ).hidden_states
        outputs = hidden_states[-1][:, -1, :]
        if outputs.dtype == torch.bfloat16:
            outputs = outputs.float()
        end = time.time()
        print(f"document {i}: {n / (end - start)} = {n} tokens / {end - start} s | Elapsed: {end - start_} s")

    print('end')
