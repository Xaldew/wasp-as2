#!/usr/bin/env python

import sys
import locale
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(csvfile):
    df = pd.read_csv(csvfile)
    classifiers = df.columns.values.tolist()
    X = np.zeros(df.dropna().shape)
    m = np.zeros(len(classifiers)) 
    s = np.zeros(len(classifiers))

    plt.figure(1)
    plt.clf()
    for k in range(len(classifiers)):
        plt.subplot(6, 2, k+1)
        X[:, k] = df[classifiers[k]].dropna().tolist()
        
        m[k] = np.mean(X[:, k])
        s[k] = np.std(X[:, k]) 

        print("{}: mu: {}, std: {}".format(classifiers[k], m[k], s[k]))

        plt.hist(X[:, k], bins=(np.arange(0, 1, 0.01) + 0.005), density=True)
        plt.title(classifiers[k])

    idx = np.argsort(m)[::-1]
    plt.figure(2)
    plt.clf()
    plt.boxplot(X[:, idx], labels=np.array(classifiers)[idx], bootstrap=10000)

    plt.show()

if __name__ == "__main__":

    fmtr = argparse.RawDescriptionHelpFormatter
    desc = "Plot MC simualtion results"
    parser = argparse.ArgumentParser(description=desc, formatter_class=fmtr)
    parser.add_argument("csvfile", metavar="FILE", type=str,
                        help="The CSV file to read")
    
    ARGS = parser.parse_args(sys.argv[1:])
    locale.setlocale(locale.LC_ALL, "")
    sys.exit(main(ARGS.csvfile))
