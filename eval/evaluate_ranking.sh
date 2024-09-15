# bm25 evaluate
for split in train test;do
    echo -ne "mdrag-5K-"${split}" | ad-hoc document ranking | baseline | bm25 | "
    ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.10 \
        ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_adhoc_dr.txt \
        retrieval/baseline.bm25.mdrag-5K-${split}.documents.run \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

    echo -ne "mdrag-5K-"${split}" | ad-hoc passage ranking  | baseline | bm25 | "
    ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.10 \
        ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_adhoc_pr.txt \
        retrieval/baseline.bm25.mdrag-5K-${split}.passages.run \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

    echo -ne "mdrag-5K-"${split}" | context passage ranking | baseline | bm25 | "
    ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.10 \
        ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
        retrieval/baseline.bm25.mdrag-5K-${split}.passages.run \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

    # contriever-MS evaluate
    echo -ne "mdrag-5k-"${split}" | ad-hoc passage ranking  | baseline | contriever | "
    ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.10 \
        ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_adhoc_pr.txt \
        retrieval/baseline.contriever.mdrag-5K-${split}.passages.run \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'

    echo -ne "mdrag-5k-"${split}" | context passage ranking | baseline | contriever | "
    ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.10 \
        ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
        retrieval/baseline.contriever.mdrag-5K-${split}.passages.run \
        | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
done
