import torch
from datasets import load_dataset 
import time
from tqdm import tqdm

dataset = load_dataset('BeIR/scidocs', 'corpus', keep_in_memory=True)['corpus']
documents = dataset['text'][:64]
guideline = "- 5: The context is highly relevant, complete, and accurate.\n- 4: The context is mostly relevant and complete but may have minor gaps or inaccuracies.\n- 3: The context is partially relevant and complete, with noticeable gaps or inaccuracies.\n- 2: The context has limited relevance and completeness, with significant gaps or inaccuracies.\n- 1: The context is minimally relevant or complete, with substantial shortcomings.\n- 0: The context is not relevant or complete at all."
template_rating = "Instruction: {INST}\n\nGuideline:\n{G}\n\nQuestion: {Q}\n\nContext: {C}\n\n{PREFIX}" 
instruction_rating = "Determine whether the question can be answered based on the provided context? Rate the context with on a scale from 0 to 5 according to the guideline below. Do not write anything except the rating."
def prompt_rating_gen(INST="", Q="", C="", PREFIX="Rating:"):
    p = template_rating
    p = p.replace("{G}", guideline)
    p = p.replace("{INST}", INST).strip()
    p = p.replace("{Q}", Q)
    p = p.replace("{C}", C)
    p = p.replace("{PREFIX}", PREFIX).strip()
    return p

prompts = list(
        prompt_rating_gen(
            INST=instruction_rating, 
            Q=' What is the central role that software development plays in the delivery',
            C=d,
        ) for d in documents
)

model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

from vllm import LLM
from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    skip_special_tokens=False,
    max_tokens=3,
)
model = LLM(
    model_name_or_path, 
    dtype='half',
    enforce_eager=True
)


with torch.no_grad():

    print("Start")
    start_ = time.time()
    for i, prompt in enumerate(prompts):
        if i == 0:
            print(prompt)

        start = time.time() 
        outputs = model.generate(prompt, sampling_params)[0].outputs[0].text
        end = time.time() 
        # print(outputs)
        print(f"document {i}: {(end - start)} | Elapsed: {end - start_} s")

    print('end')


