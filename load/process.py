from __future__ import print_function
import pandas as pd
import sys
import argparse
import os
import random
import pprint
pp = pprint.PrettyPrinter(depth=6)


def get_random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile):
      if random.randrange(num + 2): continue
      line = aline
    return line


def process(filename):
    with open(filename, 'r') as f:
        first_line = f.readline().split(',')
        random_line = get_random_line(f).split(',')

    d = {}
    for field, value in zip(first_line, random_line):
        if value != '':
            d[field] = value
    pp.pprint(d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('file', help='CSV file to load')
    args = parser.parse_args()
    process(args.file)
