# bm25 evaluate
for retrieval in bm25 contriever;do
    for split in test testb;do
    # ad-hoc ranking
        echo -ne "RACE-"${split}" | baseline | ${retrieval} | P@R_p | "
        ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.100 \
            ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_adhoc_pr.txt \
            retrieval/baseline.${retrieval}.race-${split}.passages.run \
            | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
        echo -ne "RACE-"${split}" | baseline | ${retrieval} | P@_p_min | "
        ~/trec_eval-9.0.7/trec_eval -c -m Rprec -m recall.100 \
            ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
            retrieval/baseline.${retrieval}.race-${split}.passages.run \
            | cut -f3 | sed ':a; N; $!ba; s/\n/ | /g'
    done
done
