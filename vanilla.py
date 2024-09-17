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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str, default=None)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # load topics
    topics = {}
    if os.path.isdir(args.topics):
        pass
    else:
        with open(args.topics) as f:
            for line in f:
                id, text = line.split('\t')
                topics[id] = text.strip()

    # load collection
    collection = load_collection(args.collection)

    # load run (retrieval)
    run = load_run(args.run)

    writer = open(args.output_file, 'w')

    for example_id in tqdm(topics, total=len(topics)):
        topic = topics[example_id]
        candidate_docids = [d for d in run[example_id][:args.topk]]
        candidate_docs = [collection[id] for id in candidate_docids]

        logger.info(f"Raw generated passages: {candidate_docs[-1]}")
        item = {
            'example_id': example_id, 
            'type': 'vanilla',
            'output': candidate_docs
        }
        writer.write(json.dumps(item, ensure_ascii=False)+'\n')
    writer.close()

if __name__ == '__main__':
    with torch.no_grad():
        main()

