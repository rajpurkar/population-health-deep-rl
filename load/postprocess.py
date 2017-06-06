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
import numpy as np


def filter_na_for_field(df, must_have_field):
    matching_cols = filter(lambda x: must_have_field.lower() in x.lower(), df)
    assert(len(matching_cols) == 1)
    right_col = matching_cols[0]
    df[right_col].replace('', np.nan, inplace=True)
    df = df.loc[df[right_col].notnull()]
    return df

def post_process(file, must_have_field=None, verbose=False):
    df = pd.read_csv(file, low_memory=False)
    if must_have_field is not None:
        df = filter_na_for_field(df, must_have_field)

    avoid_phrases = [
        "ID",
        "Presence",
        "age ",
        "Result of malaria measurement",
        "Number",
        "Relationship structure"
    ]

    selected = []
    for col in df:
        num_zero = sum(pd.isnull(df[col]))
        if  (
                num_zero == 0
                and len(df[col].unique()) > 1
                and len(df[col].unique()) <= 10
                and not any([phrase in col for phrase in avoid_phrases])
            ):
            selected.append(col)
            if verbose is True:
                print(col, num_zero)
    df.to_csv(os.path.dirname(file) + '/post-processed.csv',
        columns = selected,
        mode = 'w',
        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post process data.')
    parser.add_argument('file', help='File to postprocess')
    parser.add_argument('--verbose', help='verbose flag', action='store_true')
    parser.add_argument(
        '--must_have',
        help='must_have_field',
        default='Final result of malaria from blood smear test')
    args = parser.parse_args()
    post_process(args.file,
        must_have_field=args.must_have,
        verbose=args.verbose)
