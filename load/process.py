from __future__ import print_function
import pandas as pd
import sys
import argparse
import os
import random
import pprint
import shlex
import collections
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
    return fields_to_names, fields_to_parent_fields, field_to_values_to_interprets


def process(file, header_file):
    fields_to_names, fields_to_parent_fields, field_to_values_to_interprets = process_headers(header_file)
    field_to_value_for_record = get_single_record(file)
    names_to_value = {}
    parent_field_used = {}
    for field in field_to_value_for_record:
        try:
            name = fields_to_names[field]
            if field in fields_to_parent_fields:
                parent_field = fields_to_parent_fields[field]
            else:
                parent_field = field
            value = field_to_value_for_record[field].strip()
            if parent_field in field_to_values_to_interprets:
                assert(value in field_to_values_to_interprets[parent_field])
                interpret_value = field_to_values_to_interprets[parent_field][value]
            else:
                interpret_value = value
            assert(parent_field not in parent_field_used)
            assert(name not in names_to_value)
            names_to_value[name] = value
            parent_field_used[parent_field] = True
            print("{: >10} {: <60.60} {:<3}".format(parent_field, name, interpret_value))
        except:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('file', help='CSV file to load')
    parser.add_argument('header_file', help='header file to load')
    args = parser.parse_args()
    process(args.file, args.header_file)
