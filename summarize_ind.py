import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import torch
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

from retrieval_augmentation.utils import (
    batch_iterator, 
    load_model, 
    load_run, 
    load_collection
)

def truncate_and_concat(texts, tokenizer, max_length=512, offset=6):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_lengt_single_sentence-1)
    if (length+offset) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causalLM"])
    parser.add_argument("--template", type=str, default="title: {T} content: {P}")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)

    # IR requirements
    parser.add_argument("--topics", type=str, default=None)
    parser.add_argument("--rewritten", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--truncate", type=int, default=6)
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class.lower())
    model.eval()

    # load topics
    topics = {}
    if os.path.isdir(args.topics):
        pass
    else:
        with open(args.topics) as f:
            for line in f:
                id, text = line.split('\t')
                topics[id] = text.strip()

    # if args.rewritten:
    #     topics = load_rewrite(args.eval_rewrite_file)

    # load collection
    collection = load_collection(args.collection)

    # load run (retrieval)
    run = load_run(args.run)
    writer = open(args.output_file, 'w')
    all_example_ids = list(topics.keys())

    iters = batch_iterator(all_example_ids, 20) 
    for batch_example_id in tqdm(iters, total=len(all_example_ids) //20 + 1):

        # collect data
        example_ids = []
        inputs = []
        for example_id in batch_example_id:

            topic = topics[example_id]
            for id in run[example_id][:args.topk]:
                doc = truncate_and_concat(
                    collection[id], 
                    tokenizer=tokenizer,
                    max_length=args.max_length, 
                    offset=args.truncate
                )
                input = args.template.replace("{Q}", topic).replace("{P}", doc)

                example_ids.append(example_id)
                inputs.append(input)

        # batch inference
        generations = defaultdict(list)
        iterables = list(range(len(example_ids)))
        for s, e in batch_iterator(iterables, args.batch_size, return_index=True):

            batch_example_ids = example_ids[s:e]
            batch_inputs = inputs[s:e]
            tokenized_input = tokenizer(
                batch_inputs, 
                padding=True, 
                truncation=True, 
                max_length=args.max_length, 
                return_tensors='pt'
            ).to(model.device)

            outputs = model.generate(**tokenized_input, min_new_tokens=32, max_new_tokens=512)
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            if args.model_class == 'causalLM':
                outputs = [o.split('</s>')[0] for o in outputs]

            for example_id, output in zip(batch_example_ids, outputs):
                generations[example_id].append(output)

            logger.info(f"Summarizaiton-{args.model_name_or_path}: {output}")

        # write
        for example_id in generations:
            item = {
                'example_id': example_id, 
                'type': args.model_name_or_path, 
                'output': generations[example_id]
            }
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()

