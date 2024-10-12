import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import yaml
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from retrieval_augmentation.utils import load_collection

def normalize_list(string_list):
    for i in range(len(string_list)):
        string_list[i] = normalize_text(string_list[i])
    return string_list

def flatten_and_normalize(string_list):
    string = " ".join(string_list)
    return normalize_text(string)

def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

def normalize_text(string):
    string = string.strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    return string

def maybe_chunking(dlist, n=1024):
    overlength = [(i, len(d.split()) > n) for i, d in enumerate(dlist)]

    if any([o for _, o in overlength]):
        to_return = []
        for i, do_chunk in overlength:
            if do_chunk:
                words = dlist[i].split()
                while len(words) > 0:
                    to_return.append(" ".join(words[:512]))
                    words = words[512:]
            else:
                to_return.append(dlist[i])
        return to_return
    else:
        return dlist

def load_qrel(path, threshold=0):
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) >= threshold:
                data[item[0]].append( item[2] )
    return data

def main():
    parser = argparse.ArgumentParser()
    # Evaluation file is a json file that contains a list of item, each of which contains
    parser.add_argument("--multi_news_file", type=str, help="Path to multi-news")
    parser.add_argument("--duc04_file", type=str, help="Path to multi-news")
    parser.add_argument("--split", type=str, default='train', help="Original split of datasets")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    parser.add_argument("--collection", type=str, help="collection that has p_min")
    parser.add_argument("--qrel", type=str, help="qrels that has p_min")
    parser.add_argument("--output_file", type=str, help="file for the output result")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")

    # Model and name
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)") # use shard here
    args = parser.parse_args()

    # Generate prompts
    np.random.seed(args.seed)

    # Load evaluation data
    from datasets import load_from_disk
    if args.multi_news_file is not None:
        multi_news = load_from_disk(args.multi_news_file)[args.split]

        multi_news = multi_news.map(lambda x: {
            "document": normalize(x['document']), 
            'mds-source': 'multi_news'
        })
        multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )
        multi_news = multi_news.map(lambda x: {
            "document": maybe_chunking(x['document'], n=1024)
        })
        dataset = multi_news

    if args.duc04_file is not None:
        duc04 = load_from_disk(args.duc04_file)['train']
        duc04 = duc04.map(lambda x: {
            "document": normalize_list(x['context']),
            "summary": flatten_and_normalize(x['summary']),
            'mds-source': 'duc04'
        })
        duc04 = duc04.filter(lambda x: len(x['document']) >=2 )
        duc04 = duc04.map(lambda x: {
            "document": maybe_chunking(x['document'], n=1024)
        })
        dataset = duc04

    if args.qrel is not None:
        qrels = load_qrel(args.qrel)
        passages = load_collection(args.collection)

    # Sample quick test
    if args.quick_test is not None:
        np.random.seed(args.seed)
        ids = np.random.choice(len(dataset), args.quick_test, replace=False)
        dataset = [dataset[int(idx)] for idx in ids]
    else:
        if args.split == 'train':
            dataset = [dataset[idx] for idx in range(len(dataset))]
        else:
            dataset = [dataset[idx] for idx in range(min(5000, len(dataset)))]
        ids = list(range(len(dataset)))

    # Save data as ...
    writer = open(args.output_file, 'w')

    logger.info("Save dataset as vanilla baseline ...") 
    for idx, item in enumerate(tqdm(dataset)):
        example_id = f"{item['mds-source']}-{args.split}-{ids[idx]}"
        if 'report' in args.tag:
            item = {
                'example_id': example_id, 
                'type': f'oracle-report (md-summary)',
                'output': normalize_text(item['summary'])
            }

        if 'documents' in args.tag:
            item = {
                'example_id': example_id, 
                'type': f'oracle-documents',
                'output': item['document']
            }

        if 'passages' in args.tag:
            item = {
                'example_id': example_id, 
                'type': f'oracle-passages',
                'output': [passages[id] for id in qrels[example_id]]
            }

        writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

if __name__ == "__main__":
    main()

