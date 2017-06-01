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


def analyze(file):
    p_threshold = 1e-4
    min_samples = 5
    df = pd.read_csv(file, low_memory=False)
    chosen = 'Final result of malaria from blood smear test'
    df = df.loc[df[chosen].isin(['Positive', 'Negative'])]
    tups = []
    for col in df:
        if col == chosen: continue
        confusion_matrix = pd.crosstab(df[chosen], df[col])
        if (confusion_matrix > min_samples).all().all() is False: continue
        c, p, dof, elems = ss.chi2_contingency(confusion_matrix, correction=False)
        if p >= p_threshold: continue
        tups.append((col, p))
    tups = sorted(tups, key=lambda t: (t[1]))
    for tup in tups[:25]:
        print("{: >70.70} {: >5.5}".format(*tup))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to analyze')
    args = parser.parse_args()
    analyze(args.file)
