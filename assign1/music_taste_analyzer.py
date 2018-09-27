#!/usr/bin/env python

import sys
import random
import pickle
import locale
import argparse
import warnings
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier


CLASSIFIERS = {
    "knn": KNeighborsClassifier(),
    "svm": SVC(),
    "nusvm": NuSVC(),
    "dtree": DecisionTreeClassifier(),
    "rdforest": RandomForestClassifier(),
    "adaboost": AdaBoostClassifier(),
    "grdboost": GradientBoostingClassifier(),
    "nbayes": GaussianNB(),
    "gaussproc": GaussianProcessClassifier(),
    "lda": LinearDiscriminantAnalysis(),
    "qda": QuadraticDiscriminantAnalysis(),
    "mlpc": MLPClassifier(),
}

TUNING = {
    "knn": [
        {
            "n_neighbors": range(1, 20),
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3, 4]
        },
    ],
    "svm": [
        {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
        {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]},
        {
            "C": [0.1, 0.5, 1, 10, 100],
            "kernel": ["poly"],
            "degree": range(1, 10),
            "coef0": [0.1, 0.5, 1.0],
        },
    ],
    "nusvm": [
        {},
    ],
    "dtree": [{}],
    "rdforest": [{}],
    "adaboost": [{}],
    "grdboost": [{}],
    "nbayes": [{}],
    "gaussproc": [{}],
    "lda": [{}],
    "qda": [{}],
    "mlpc": [{}],
}

RD_TUNING = {
    "knn":
    {
        "n_neighbors": range(1, 100),
        "weights": ["uniform", "distance"],
        "p": range(1, 100)
    },
    "svm":
    {
        "C": scipy.stats.expon(scale=100),
        "kernel": ["linear", "rbf", "poly"],
        "gamma": scipy.stats.expon(scale=.1),
        "class_weight": ["balanced", None],
        "coef0": scipy.stats.expon(scale=1.0),
        "degree": range(1, 10),
    },
    "nusvm":
    {
        "nu": scipy.stats.uniform(0.0, 1.0),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": range(1, 10),
        "gamma": scipy.stats.expon(scale=.1),
        "coef0": scipy.stats.expon(scale=1.0),
    },
    "dtree":
    {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_depth": range(4, 16),
        "max_features": [None, "auto", "sqrt", "log2"],
    },
    "rdforest":
    {
        "n_estimators": range(5, 100),
        "criterion": ["gini", "entropy"],
        "max_depth": range(4, 16),
        "max_features": [None, "auto", "sqrt", "log2"],
    },
    "adaboost":
    {
        "n_estimators": range(30, 100),
        "learning_rate": scipy.stats.uniform(0.0, 10.0),
        "algorithm": ["SAMME", "SAMME.R"],
    },
    "grdboost":
    {
        "loss": ["deviance", "exponential"],
        "learning_rate": scipy.stats.uniform(0.0, 1.0),
        "n_estimators": range(100, 10000, 500),
        "max_depth": range(2, 16),
        "criterion": ["friedman_mse"],
    },
    "nbayes": {},
    "gaussproc":
    {
        "warm_start": [False, True],
        "n_restarts_optimizer": range(1, 10),
        "max_iter_predict": range(100, 1000, 100),
    },
    "lda":
    {
        "solver": ["svd", "lsqr", "eigen"],
    },
    "qda": {},
    "mlpc":
    {
        "hidden_layer_sizes": [(100, 50, 25),
                               (500, 250, 100, 25, 10, 5),
                               (50, 20, 10),
                               (30, 20, 10, 5, 2),
                               (30, 10, 5),
                               (30, 10),
                               (10, 5)],
        "alpha": scipy.stats.uniform(0.0, 1.0),
        "activation": ["identity", "logistic", "tanh", "relu"],
    },
}

FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "tempo",
    "valence",
]


PARAM_SPACE_SZ = {
    "knn": float("inf"),
    "svm": float("inf"),
    "nusvm": float("inf"),
    "dtree": 2 * 2 * 12 * 4,
    "rdforest": 95 * 2 * 12 * 4,
    "adaboost": float("inf"),
    "grdboost": float("inf"),
    "nbayes": 1,
    "gaussproc": 2 * 10 * 10,
    "lda": 3,
    "qda": 1,
    "mlpc": float("inf"),
}


def main(training_data, new_data, classifier,
         seed, percent, rand, niters,
         dump="", load=""):
    """Use the selected classifier to estimate someones music taste.

    .. Returns:
    :returns: 0 if the script ran successfully, otherwise a non-zero value.
    :rtype: An integer.

    """
    training = pd.read_csv(training_data, sep=",")
    test = pd.read_csv(new_data, sep=",")

    X_train = training.loc[:,FEATURES].values
    y_train = training.loc[:,"label"].values
    X_test = test.loc[:,FEATURES].values

    # Can also be done using sklearn methods such as MinMaxScaler().
    X_trainn = X_train*1/np.max(np.abs(X_train),axis=0)
    X_testn = X_test*1/np.max(np.abs(X_train),axis=0)

    # Note: All inputs/features are treated as quantitative/numeric some of the
    # features are perhaps more sensible to treat as
    # qualitative/cathegorical. For that sklearn preprocessing methods such as
    # OneHotEncoder() can be used.

    # Split into training/validation set.
    x_train, x_val, y_train, y_val = train_test_split(X_trainn, y_train,
                                                      test_size=percent,
                                                      random_state=seed)

    # Feed it with data and train it and write out score.
    clf = None
    niters = min(PARAM_SPACE_SZ[classifier], niters)
    if rand:
        clf = RandomizedSearchCV(CLASSIFIERS[classifier], RD_TUNING[classifier],
                                 cv=5, n_jobs=-1, n_iter=niters,
                                 error_score=0,
                                 random_state=seed)
    else:
        clf = GridSearchCV(CLASSIFIERS[classifier], TUNING[classifier],
                           cv=5, n_jobs=-1)

    # Override and load model from the given file.
    if load:
        with open(load, "r") as f:
            clf = pickle.load(f)

    # Do the actual training.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not load:
            clf.fit(X=x_train, y=y_train)

    print("%s parameters: %s" % (classifier, str(clf.best_params_)))
    print("%s search score: %.2f %%" % (classifier, 100.0 * clf.best_score_))
    print("%s validation score: %.2f %%" % (classifier, 100.0 * clf.score(x_val, y_val)))

    # Dump the trained model.
    if dump:
        with open(dump, "w") as f:
            pickle.dump(clf, f)

    # Compute the prediction on the test and print the labels as a single line.
    predictions = (clf.predict(X=X_testn)
                   .reshape(-1,1)
                   .astype(int)
                   .reshape(1,-1))
    print("#Ones: %d" % sum(predictions.flatten()))
    print(" ".join(str(i) for i in predictions.flatten()))

    return 0


def parse_arguments(argv):
    """Parse the given argument vector.

    .. Keyword Arguments:
    :param argv: The arguments to be parsed.

    .. Types:
    :type argv: A list of strings.

    .. Returns:
    :returns: The parsed arguments.
    :rtype: A argparse namespace object.

    """
    fmtr = argparse.RawDescriptionHelpFormatter
    kdesc = "Music Taste Analyzer"
    parser = argparse.ArgumentParser(description=kdesc, formatter_class=fmtr)
    parser.add_argument("training_data", metavar="FILE", type=str,
                        help="The Music Taste classifier training data.")
    parser.add_argument("new_data", metavar="FILE", type=str,
                        help="The new songs to test the classifier on.")
    parser.add_argument("-c", "--classifier", action="store",
                        choices=CLASSIFIERS.keys(), type=str, default="knn",
                        help="The type of classifier to use.")
    parser.add_argument("-s", "--seed", action="store", type=int, default=2,
                        help="The seed to use for splitting the training data.")
    parser.add_argument("-p", "--percent", action="store",
                        type=float, default=0.2,
                        help="Percent of the training/validation splits.")
    parser.add_argument("-r", "--random", action="store_true",
                        help=("Perform a randomized search over the "
                              "hyperparameters."))
    parser.add_argument("-n", "--n-iters", action="store",
                        type=int, default=10,
                        help="Number of iterations for the randomized search.")
    parser.add_argument("-d", "--dump", action="store", type=str,
                        help="File to dump the classifier model to.")
    parser.add_argument("-l", "--load", action="store", type=str,
                        help="Load the classifier model from this file.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ARGS = parse_arguments(sys.argv[1:])
    locale.setlocale(locale.LC_ALL, "")
    random.seed(ARGS.seed)
    sys.exit(main(ARGS.training_data,
                  ARGS.new_data,
                  ARGS.classifier,
                  ARGS.seed,
                  ARGS.percent,
                  ARGS.random,
                  ARGS.n_iters,
                  ARGS.dump,
                  ARGS.load))
