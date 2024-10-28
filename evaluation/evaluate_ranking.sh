# bm25 evaluate
for split in test testb;do
    echo "RACE-"${split}

for retrieval in bm25 contriever splade;do
    printf '%-12s|' ' baseline' ${retrieval}
    ir_measures \
        ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        retrieval/all_corpus/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec(rel=1) RPrec(rel=3) R(rel=3)@10 R(rel=3)@20' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
done
done
