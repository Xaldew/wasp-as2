#!/usr/bin/env python

import pandas as pd

def main(csvfile, classifiers):
    df = pd.DataFrame([[None]*len(classifiers)], columns=classifiers)
    df.to_csv(csvfile, index=False)  

if __name__ == "__main__":
    ARGS = sys.argv[1:]
    sys.exit(main(sys.argv[1], sys.argv[2:])) 
