import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import re
import os
import argparse
import json
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default=None, help="File path to the file with generated texts.")
    parser.add_argument("--output_dir", type=str, default=None, help="Dir to the output collection.")
    args = parser.parse_args()

    split = 'train'
    if 'test' in args.dataset_file:
        split = 'test'
    if 'testb' in args.dataset_file:
        split = 'testb'
    passages = {}
    documents = {}

    # Load context
    with open(args.dataset_file, 'r') as f:
        for line in tqdm(f, "load contexts"):
            data = json.loads(line.strip())
            example_id = data['example_id']
            documents_ = data['documents']
            passages_ = data['passages']
            n_passages = sum(len(plist) for plist in passages_)
            questions, ratings = data['questions'], data['ratings']
            ratings = np.array(ratings)

            # sanity check
            if ratings.shape != (n_passages, len(questions)):
                logging.warnings(f"example id: {example_id} has incorrect number of passages: {n_passages} ({len(questions)} questions).")
                continue

            if len(documents_) != len(passages_):
                logging.warnings(f"example id: {example_id} has inconsistent number of context: #d: {len(documents_)} #p: {len(passages_)}.")
                continue

            j = 0
            for i, plist in enumerate(passages_):
                docid = f"{example_id}:{i}"
                documents[docid] = documents_[i]

                for passage in plist:
                    psg_id = f"{docid}#{j}"
                    passages[psg_id] = passage
                    j += 1

    # Save the context
    output_dir = os.path.join(args.output_dir, 'documents')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}_docs.jsonl")
    with open(output_file, 'w') as f:
        for docid, document in documents.items():
            f.write(json.dumps({"id": docid, "contents": document}, ensure_ascii=False)+'\n')

    output_dir = os.path.join(args.output_dir, 'passages')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{split}_psgs.jsonl")
    with open(output_file, 'w') as f:
        for psgid, passage in passages.items():
            f.write(json.dumps({"id": psgid, "contents": passage}, ensure_ascii=False)+'\n')

    print("done")
