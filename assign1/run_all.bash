#!/usr/bin/env bash
# Run all classifiers, saving the models.

models=(
    # "knn"
    # "svm"
    # "nusvm"
    # "dtree"
    "rdforest"
    # "adaboost"
    "grdboost"
    # "nbayes"
    # "gaussproc"
    # "lda"
    # "qda"
    # "mlpc"
)

for mod in ${models[@]};
do
    printf "Running: ${mod}\n"
    python ./music_taste_analyzer.py training_data.csv songs_to_classify.csv \
           --classifier "${mod}" \
           --seed ${1:-2} \
           --random \
           --percent 0.2 \
           --n-iters 100 \
           --dump ${mod}.pkl
done
