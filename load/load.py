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
        convert_missing=False)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(100*1000)
        while len(chunk) > 0:
            df = df.append(chunk, ignore_index=True)
            chunk = reader.get_chunk(100*1000)
            print('.'),
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print('\nloaded {} rows'.format(len(df)))
    return df


def convert_to_csv(dta, filename):
    print("saving to csv...")
    dta.to_csv(filename + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load data.')
    parser.add_argument('file',
                        help='File to load')
    parser.add_argument('--output_dir',
                        help='Output dir', default=".")
    args = parser.parse_args()
    dta = load_large_dta(args.file)
    fname = os.path.basename(args.file)
    convert_to_csv(dta, args.output_dir + '/' + fname)