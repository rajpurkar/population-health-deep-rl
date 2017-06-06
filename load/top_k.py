from __future__ import print_function
import pandas as pd
import sys
import argparse
import os
import random
import pprint
import shlex
import collections
import glob
import csv
from tqdm import tqdm
import pandas as pd
import scipy.stats as ss
import numpy as np

def get_correlations(df, output_field):
    p_threshold = 1e-4
    min_samples = 10
    tups = []
    for col in df:
        if col == output_field: continue
        confusion_matrix = pd.crosstab(df[col], df[output_field])
        if (confusion_matrix > min_samples).all().all() is False: continue
        c, p, dof, elems = ss.chi2_contingency(confusion_matrix, correction=False)
        if p >= p_threshold: continue
        tups.append((col, p, confusion_matrix))
    return tups


def get_top_k_correlated_cols(df, output_field, k, verbose=True):
    tups = get_correlations(df, output_field)
    print(k)
    tups = sorted(tups, key=lambda t: (t[1]))[:k]
    if verbose is True:
        for (col, p, confusion_matrix) in tups:
            print("{: <70.70} {: >5.5}".format(col, p))
            print(confusion_matrix)
    cols = map(lambda x: x[0], tups)
    return cols


def correlate(file, output_field, k):
    df = pd.read_csv(file, low_memory=False)
    cols = get_top_k_correlated_cols(df, output_field, k)
    cols.append(output_field)
    df.to_csv(os.path.dirname(file) + '/top-' + str(k) + '.csv',
        columns = cols,
        mode = 'w',
        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to analyze')
    parser.add_argument('-k',
        help='Top k',
        default=10,
        type=int)
    parser.add_argument('--output_field',
        help='Output field',
        default='Final result of malaria from blood smear test')
    args = parser.parse_args()
    correlate(args.file, args.output_field, args.k)
