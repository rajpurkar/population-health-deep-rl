from __future__ import print_function
import pandas as pd
import sys
import argparse
import os


def load_large_dta(fname):
    reader = pd.read_stata(
        fname,
        iterator=True,
        convert_dates=False,
        convert_categoricals=False,
        convert_missing=False,
        chunksize=100000)
    df = pd.DataFrame()

    try:
        for chunk in reader:
            df = df.append(chunk, ignore_index=True)
            print('.'),
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print('\nloaded {} rows'.format(len(df)))
    return df


def convert_to_csv(dta, filename):
    print("saving to csv...")
    dta.to_csv(filename + '.CSV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load data.')
    parser.add_argument('file', help='File to load')
    args = parser.parse_args()
    dta = load_large_dta(args.file)
    convert_to_csv(dta, args.file)