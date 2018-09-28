#!/usr/bin/env bash
# Run all classifiers a set amout of times, saving the progress of each in each
# file. Warning: will overwrite any file already present at csvfile argument

csvfile="data/log.csv"
simulations=200
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

# Prepare the csv file
python ./init_logger.py ${csvfile} "${models[@]}"

# Run the simulation
for sim in $(seq ${simulations})
do
    for mod in ${models[@]};
    do
        printf "Running: ${mod}\n"
        python ./music_taste_analyzer.py training_data.csv songs_to_classify.csv \
               --classifier "${mod}" \
               --random \
               --seed $RANDOM \
               --percent 0.2 \
               --n-iters 100 \
               --dump ${mod}.pkl \
               --file ${csvfile} >> data/output.txt
    done
done
