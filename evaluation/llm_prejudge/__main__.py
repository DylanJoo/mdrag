import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import re
import argparse
import json
import numpy as np
from copy import copy
import random
from collections import defaultdict
from tqdm import tqdm
from glob import glob

from transformers import AutoTokenizer

def load_qrel(path, threshold=0):
    data = defaultdict(list)
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            item = line.strip().split()
            if int(item[3]) >= threshold:
                data[item[0]].append( item[2] )
    return data

def load_judgements(path, report_file=None):
    judgements = defaultdict(lambda: defaultdict(lambda: None))
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data['example_id']
            judgements[example_id].update({data['pid']: data['rating']})

    if report_file is not None:
        with open(report_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                example_id = data['example_id']
                judgements[example_id].update({f'{example_id}_report': data['rating']})

    return judgements

def load_run(path, topk=9999):
    run = defaultdict(list)
    if path is None:
        return None
    with open(path, 'r') as f:
        for line in f:
            example_id, Q0, psgid, rank, score, prefix = line.strip().split()
            if int(rank) <= topk:
                run[example_id].append(psgid)
    return run

def load_passages(dir, report_file=None):
    passages = {}
    for file in glob(os.path.join(dir, "*jsonl")):
        with open(file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                passages[item["id"]] = item['contents']

    if report_file is not None:
        with open(report_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                example_id = item['example_id']
                passage[example_id] = item['contents']
    return passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print output")
    parser.add_argument("--generator_name", type=str, default=None)
    parser.add_argument("--run_file", type=str, default=None)
    parser.add_argument("--topk", type=int, default=1000)

    parser.add_argument("--judgement_file", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument("--n_questions", type=int, default=None)

    parser.add_argument("--qrels", type=str, default=None, help="File path to the qrels.")
    parser.add_argument("--rel_threshold", type=float, default=1)

    parser.add_argument("--report_file", type=str, default=None)
    parser.add_argument("--passage_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.generator_name)

    # load qrels, pre-calculated judge
    qrels = load_qrel(args.qrels, threshold=args.rel_threshold)
    runs = load_run(args.run_file, args.topk)
    judgements = load_judgements(args.judgement_file, report_file=args.report_file)
    passages = load_passages(args.passage_dir, report_file=args.report_file)

    # load graded passages
    outputs = {'coverage': [], 'density': [], 'num_segs': [], 'num_tokens': []}

    # oracle-report
    for example_id in qrels:

        # maximum of answerable questions
        # n_questions = ( len(judgements[example_id]['report']) or args.n_questions)
        n_questions = args.n_questions

        if 'oracle-report' in args.tag:
            psgids = [f'{example_id}_report']
        if 'oracle-passages' in args.tag:
            psgids = qrels[example_id]
        else:
            psgids = runs[example_id]

        # collection prejudged ratings
        context = ""
        ratings = [[0] * n_questions]
        for psgid in psgids:
            context = context + " " + passages[psgid]
            if (example_id == psgid.split(":")[0]) or (psgid == 'report'):
                judgement = judgements[example_id][psgid]
                ratings.append(judgement)


        # get maximun ratings
        ratings = np.array(ratings).max(0)
        n_answerable = sum(ratings >= args.threshold)

        ## calculate coverage
        coverage = n_answerable / n_questions
        outputs['coverage'].append(coverage)

        ## calculate density
        n_tokens = len(tokenizer.tokenize(context))
        density = coverage / n_tokens
        outputs['density'].append(density)

        outputs['num_segs'].append(len(psgids))
        outputs['num_tokens'].append(n_tokens)


    # results
    mean_coverage = np.mean(outputs['coverage'])
    mean_density = np.mean(outputs['density']) * 100
    mean_num_segments = np.mean(outputs['num_segs'])
    mean_num_tokens = np.mean(outputs['num_tokens'])
    print(f" # === Evaluation Results === ")
    print(f" # TAG : {args.tag.replace('topk', str(args.topk))} | {len(outputs['coverage'])} examples")
    print(f' # Mean Coverage (tau={args.threshold})  : {mean_coverage:.4f}')
    print(f' # Mean Density % (tau={args.threshold}) : {mean_density:.4f}')
    print(f' # Mean number of segments  : {mean_num_segments:.2f}')
    print(f' # Mean number of tokens    : {mean_num_tokens:.2f}\n')
