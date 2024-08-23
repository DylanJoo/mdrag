# Load evaluation data
from datasets import load_from_disk, concatenate_datasets
multi_news = load_from_disk("/home/dju/datasets/multi_news")['train']

# Preproces dataset
import re
def normalize(string):
    string = string.strip()
    pattern = re.compile(r"\s+")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile(r"\n")
    string = re.sub(pattern, ' ', string).strip()
    pattern = re.compile("</s>")
    string = re.sub(pattern, '|||||', string).strip() # align seperation 
    return string.split('|||||')

multi_news = multi_news.map(lambda x: 
    {"document": normalize(x['document']), 'mds-source': 'multi_news'}
)
multi_news = multi_news.filter(lambda x: len(x['document']) >=2 )

## chunking the long document
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

multi_news = multi_news.map(
    lambda x: {"document": maybe_chunking(x['document'], n=1024)}
)
dataset = multi_news
print(dataset[0])

n = 0
for d in dataset:
    n += len(d['document'])
print(n)
