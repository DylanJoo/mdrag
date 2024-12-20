import re
import os
import yaml
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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

    ids = list(range(len(dataset)))

    # Save data as ...
    writer = open(args.output_file, 'w')

    for idx, item in enumerate(tqdm(dataset)):
        example_id = f"{item['mds-source']}-{args.split}-{ids[idx]}"
        item = {
            'id': f"{example_id}:report", 
            'contents': normalize_text(item['summary'])
        }
        # only report has to rerun the exps
        writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

if __name__ == "__main__":
    main()

