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


def post_process(file, must_have_field=None, verbose=False):
    df = pd.read_csv(file, low_memory=False)
    if must_have_field is not None:
        matching_cols = filter(lambda x: must_have_field.lower() in x.lower(), df)
        assert(len(matching_cols) == 1)
        right_col = matching_cols[0]
        df = df.loc[df[right_col].notnull()]

    selected = []
    for col in df:
        num_zero = sum(pd.isnull(df[col]))
        if  (
                num_zero < (len(df) / 2.0)
                and len(df[col].unique()) > 1
                and len(df[col].unique()) <= 10):
            selected.append(col)
            if verbose is True:
                print(col, num_zero)
    df.to_csv(file + '-postprocessed.csv',
        columns = selected,
        mode = 'w',
        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post process data.')
    parser.add_argument('file', help='File to postprocess')
    parser.add_argument('--verbose', help='verbose flag', action='store_true')
    parser.add_argument('--must_have', help='must_have_field')
    args = parser.parse_args()
    post_process(args.file,
        must_have_field=args.must_have,
        verbose=args.verbose)
