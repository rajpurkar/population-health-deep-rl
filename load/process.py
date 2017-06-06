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
pp = pprint.PrettyPrinter(depth=6)

def process_metadata(filename):
    fields_to_names = {}
    fields_to_parent_fields = {}
    field_value_interpret = collections.defaultdict(dict)
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
                    field_value_interpret[field][line[0]] = line[1]
    return (
        fields_to_names,
        fields_to_parent_fields,
        field_value_interpret)


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


def interpret_val(
        field,
        fields_to_parent_fields,
        value,
        field_value_label):
    parent_field = get_parent_field(
                field, fields_to_parent_fields)

    if value == '': return value
    interpret_value = value
    if parent_field in field_value_label:
        try:
            value = str(int(float(value)))
        except:
            value = value
        if (value in field_value_label[parent_field]):
            interpret_value = field_value_label[parent_field][value]
    return interpret_value


def filter_headers(
        headers,
        fields_to_types,
        fields_to_parent_fields):
    new_headers = []
    for field in headers:
        assert(field == field.strip())
        if field == '': continue
        field_type = fields_to_types[field]
        if (field_type != 'byte'): continue
        parent_field = get_parent_field(
            field, fields_to_parent_fields)
        if (parent_field == field): continue
        new_headers.append(field)
    return new_headers


def convert_field_to_name(field, fields_to_names):
    if field in fields_to_names:
        return fields_to_names[field]
    else:
        return field


def load_files(data_folder):
    try:
        csv_file = glob.glob(data_folder + '/*.CSV')[0]
        do_file = glob.glob(data_folder + '/*.DO')[0]
    except:
        print("Make sure folder has a CSV, DO, and DCT file")
        raise
    try:
        dct_file = glob.glob(data_folder + '/*.DCT')[0]
    except:
        dct_file = None
    return csv_file, do_file, dct_file


def load_structures(data_folder):
    csv_file, do_file, dct_file = load_files(data_folder)
    fields_to_types = {}
    if dct_file is not None:
        fields_to_types = process_types(dct_file)
    (fields_to_names,
    fields_to_parent_fields,
    field_to_values_to_label) = process_metadata(do_file)
    reader = csv.DictReader(open(csv_file, 'r'))
    return (reader, 
        fields_to_names,
        fields_to_parent_fields,
        field_to_values_to_label,
        fields_to_types)


def process(data_folder, verbose=False):
    (reader, 
        fields_to_names,
        fields_to_parent_fields,
        field_to_values_to_label,
        fields_to_types) = load_structures(data_folder)
    headers = reader.fieldnames
    filtered_headers = filter_headers(
        headers,
        fields_to_types,
        fields_to_parent_fields)
    named_headers = [convert_field_to_name(
        field, fields_to_names) for field in filtered_headers]
    writer = csv.DictWriter(
        open(data_folder + '/processed.csv', 'w+'),
        fieldnames=named_headers)
    writer.writeheader()
    for row in tqdm(reader):
        output_row = {}
        for field in filtered_headers:
            name = convert_field_to_name(field, fields_to_names)
            value = row[field].strip()
            interpret_value = interpret_val(
                field,
                fields_to_parent_fields,
                value,
                field_to_values_to_label)
            output_row[name] = interpret_value
            if verbose is True:
                print("{: >10.10} {: >50.50} {:<15.15}"
                    .format(field, name, interpret_value))
        writer.writerow(output_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('data_folder', help='data folder to load')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    process(args.data_folder, args.verbose)
