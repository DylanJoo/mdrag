# bm25 evaluate
for split in test testb;do
    echo "RACE-"${split}

for retrieval in bm25 contriever splade;do
    echo -ne "  baseline | ${retrieval} | "
    ir_measures \
        ${DATASET_DIR}/RACE/ranking_3/70b/${split}_qrels_oracle_context_pr.txt \
        retrieval/baseline.${retrieval}.race-${split}.passages.run \
        'RPrec(rel=1) RPrec(rel=2) RPrec(rel=3) R(rel=3)@10' | cut -f2 | sed ':a; N; $!ba; s/\n/ | /g'
done
done
