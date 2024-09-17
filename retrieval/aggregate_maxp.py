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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.run, 'r') as f:
        for line in f:
            line.split()

