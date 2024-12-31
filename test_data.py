from datasets import load_dataset
dataset = load_dataset('json', data_files='data/mds-5k-greedy-1.jsonl', keep_in_memory=True)
dataset = dataset.filter(lambda x: len(x['docids']) !=0 )['train']
n_examples = len(dataset)
print(dataset)

from transformers import AutoTokenizer
from utils import load_model
model_name_or_path = 'hyen/CEPED-LLaMA-2-Chat-7B'
tokenizer, model = load_model(model_name_or_path, 'cepe')

from data.collator import Standard
data_collator = Standard(
    tokenizer=tokenizer, 
    max_src_length=1024,
    max_tgt_length=1024,
    max_num_contexts=10,
    num_distractor_docs=5,
    num_redundant_docs=1,
    shuffle=True,
)

p = data_collator([dataset[0], dataset[1]])

print(p)
