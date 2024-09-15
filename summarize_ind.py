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

def truncate_and_concat(texts, tokenizer, max_length=512):
    tokenized = tokenizer.tokenize(texts)
    length = len(tokenizer.tokenize(texts))
    max_length = (max_length or tokenizer.max_lengt_single_sentence-1)
    if (length+6) < max_length:
        return texts
    else:
        return tokenizer.convert_tokens_to_string(tokenized[:(max_length-6)])

def load_rewrite(file):
    data_items = json.load(open(file))
    rewritten_queries = {}
    for item in data_items:
        id = str(item['requestid']) + item['colectionids'].replace('neuclir/1/', '')[:2]
        rewritten_queries[id] = item['rewrite']
    return rewritten_queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_class", type=str, default=None, choices=["fid", "seq2seq", "causualLM"])
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
    parser.add_argument("--truncate", default=False, action='store_true')
    args = parser.parse_args()

    # load model
    model, tokenizer = load_model(args.model_name_or_path, model_class=args.model_class)
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

    for example_id in tqdm(topics, total=len(topics)):

        topic = topics[example_id]
        candidate_docids = [d for d in run[example_id][:args.topk]]

        # batch inference
        summaries = []
        for batch_docs in batch_iterator(candidate_docids, args.batch_size):

            batch_docs = [collection[id] for id in batch_docs]

            if args.truncate:
                batch_docs = [truncate_and_concat(doc, tokenizer, max_length=args.max_length) for doc in batch_docs]

            # independently input
            input = list(
                args.template.replace("{Q}", topic).replace("{P}", doc) \
                        for doc in batch_docs
            )
            tokenized_input = tokenizer(input, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt').to(model.device)
            outputs = model.generate(
                **tokenized_input, min_new_tokens=32, max_new_tokens=512
            )
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            summaries.extend(outputs)

        logger.info(f"Summarizaiton-{args.model_name_or_path}: {outputs[0]}")
        item = {
            'example_id': example_id, 
            'type': args.model_name_or_path, 
            'output': summaries
        }
        writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()

