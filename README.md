# Retrieval-augmented Context Evaluation

## Checkpoints 
0. Vanilla prompting. 
1. Reorgnize text-level contexts

## Results

## (Optional) Report-anchored Generation
We have already generated and released entire datasets. But you can generate the same one with the following steps.

- Download source datasets. 
There might be some different version of data. The data we used are from Huggingface. (Multi-news/TREC DUC-2004). You can also download the raw data [here]()

- Generate with Llama3.1 
    * Passages
    * Questions
    * Topics
    * Ratings

- Create collections
We have document collections (by aggregating oracle docs) as well as passage collections.
``
python create_collections.sh
``

- Indexing

- Create datasets



