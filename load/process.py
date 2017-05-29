from __future__ import print_function
import pandas as pd
import sys
import argparse
import os
import random
import pprint
import shlex
pp = pprint.PrettyPrinter(depth=6)


def get_random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile):
      if random.randrange(num + 2): continue
      line = aline
    return line


def get_single_record(data_file):
    field_to_value_for_record = {}

    with open(data_file, 'r') as f:
        first_line = f.readline().split(',')
        random_line = get_random_line(f).split(',')

    for field, value in zip(first_line, random_line):
        if value != '':
            field_to_value_for_record[field] = value
    return field_to_value_for_record


def process(file, header_file):
    field_to_value_for_record = get_single_record(file)
    fields_to_names = get_field_names(header_file)
    names_to_value = {}
    for field in field_to_value_for_record:
        try:
            name = fields_to_names[field]
            names_to_value[name] = field_to_value_for_record[field]
        except:
            continue
    pp.pprint(names_to_value)


def get_field_names(filename):
    fields_to_names = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = shlex.split(line)
            if len(line) > 1 and line[1] == 'variable':
                fields_to_names[line[2]] = line[3]
    return fields_to_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('file', help='CSV file to load')
    parser.add_argument('header_file', help='header file to load')
    args = parser.parse_args()
    process(args.file, args.header_file)
