import json
from datasets import load_from_disk, concatenate_datasets

def normalize_texts(texts, size=10000):
    texts = remove_citations(texts)
    texts = texts.strip()
    pattern = re.compile(r"\n")
    texts = re.sub(pattern, ' ', texts).strip()
    pattern = re.compile(r"\s+")
    texts = re.sub(pattern, ' ', texts).strip()
    texts = maybe_truncation(texts, size)
    return texts

def load_passages(path, n=3):
    data = json.load(open(path, 'r'))

    passages = []
    for i, item in enumerate(data['data']):
        example_id = item['example_id']

        doc_outputs = []
        for doc_output in item['docs']['output']:
            doc_output = normalize_texts(doc_output)

            if doc_output == " ":
                doc_outputs.append(["No content."])
            else:
                doc_output = doc_output.strip().split('</p>')[:n]
                doc_output = [replace_tags(o, 'p').strip() for o in doc_output]
                doc_output = [o.strip() for o in doc_output if o.strip() != ""]
                doc_outputs.append(doc_output)

        passages.append({
            "example_id": example_id, 
            "texts": doc_outputs, 
            "docs_full_texts": [normalize_texts(d) for d in item["docs"]["full_text"]]
        })
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the config file")
    parser.add_argument("--multi_news_file", type=str, default=None, help="Path to the multinews file")
    args = parser.parse_args()

    ## the collection include all the splits
    multi_news = load_from_disk(args.multi_news_file)

    for split in multi_news:
        multi_news[split]

    logger.info("load passages...") 
    passages_all = []

    for file in tqdm(glob(os.path.join(args.input_dir, "*summ*.json"))):
        passages = load_passages(file)
        passages_all += passages
    passages_all = {p['example_id']: p['texts'] for p in passages_all}

if __name__ == "__main__":
    main()

