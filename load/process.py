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
        value = value.strip()
        field = field.strip()
        if value != '':
            field_to_value_for_record[field] = value
    return field_to_value_for_record


def process_headers(filename):
    fields_to_names = {}
    fields_to_parent_fields = {}
    field_to_values_to_interprets = collections.defaultdict(dict)
    with open(filename, 'r') as f:
        lines = iter(f.readlines())
        while True:
            line = next(lines, None)
            if line is None:
                break
            line = shlex.split(line)
            if len(line) > 3 and line[1] == 'variable':
                fields_to_names[line[2]] = line[3]
            elif len(line) > 3 and line[1] == 'values':
                fields_to_parent_fields[line[2]] = line[3]
            elif len(line) > 2 and line[1] == 'define':
                field = line[2]
                while True:
                    line = next(lines, None)
                    if line is None or line == ';\r\n':
                        break
                    line = shlex.split(line)
                    field_to_values_to_interprets[field][line[0]] = line[1]
    return (
        fields_to_names,
        fields_to_parent_fields,
        field_to_values_to_interprets)


def process_types(dct_file):
    fields_to_types = {}
    with open(dct_file, 'r') as f:
        for line in f.readlines():
            line = shlex.split(line)
            if len(line) >= 3:
                fields_to_types[line[1]] = line[0]
    return fields_to_types

def get_parent_field(field, fields_to_parent_fields):
    if field in fields_to_parent_fields:
        parent_field = fields_to_parent_fields[field]
    else:
        parent_field = field
    return parent_field


def get_interpret_value(field, value, field_value_label):
    interpret_value = value
    if field in field_value_label:
        value = str(int(float(value)))
        if (value in field_value_label[field]):
            interpret_value = field_value_label[field][value]
    return interpret_value


def process(csv_file, do_file, dct_file, only_interpretable=False):
    fields_to_types = process_types(dct_file)
    (fields_to_names,
    fields_to_parent_fields,
    field_to_values_to_label) = process_headers(do_file)
    field_to_value_for_record = get_single_record(csv_file) 
    for field in field_to_value_for_record:
        if field == '': continue
        value = field_to_value_for_record[field]
        name = fields_to_names[field]
        field_type = fields_to_types[field]
        parent_field = \
            get_parent_field(field, fields_to_parent_fields)
        interpret_value = \
            get_interpret_value(parent_field, value, field_to_values_to_label)
        if only_interpretable is True:
            if interpret_value == value:
                continue
        print("{: >10.10} {: <5.5} {: >50.50} {:<15.15}"
            .format(field, field_type, name, interpret_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('data_folder', help='data folder to load')
    parser.add_argument('--interpretable', action='store_true')
    args = parser.parse_args()
    try:
        csv_file = glob.glob(args.data_folder + '/*.CSV')[0]
        do_file = glob.glob(args.data_folder + '/*.DO')[0]
        dct_file = glob.glob(args.data_folder + '/*.DCT')[0]
    except:
        print("Make sure folder has a CSV, DO, and DCT file")
    process(csv_file, do_file, dct_file, only_interpretable=args.interpretable)
