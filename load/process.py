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
    return (fields_to_names, fields_to_parent_fields, field_to_values_to_interprets)


def process_types(dct_file):
    fields_to_types = {}
    with open(dct_file, 'r') as f:
        for line in f.readlines():
            line = shlex.split(line)
            if len(line) >= 3:
                fields_to_types[line[1]] = line[0]
    return fields_to_types


def process(csv_file, do_file, dct_file):
    fields_to_types = process_types(dct_file)
    (fields_to_names,
    fields_to_parent_fields,
    field_to_values_to_interprets) = process_headers(do_file)
    field_to_value_for_record = get_single_record(csv_file)
    for field in field_to_value_for_record:
        try:
            if field == '': continue
            field = field.strip()
            assert(field in fields_to_names)
            name = fields_to_names[field]
            assert(field in fields_to_types)
            field_type = fields_to_types[field]
            if field in fields_to_parent_fields:
                parent_field = fields_to_parent_fields[field]
            else:
                parent_field = field
            value = field_to_value_for_record[field].strip()
            try:
                if parent_field in field_to_values_to_interprets:
                    value = str(int(float(value)))
                    assert(value in field_to_values_to_interprets[parent_field])
                    interpret_value = field_to_values_to_interprets[parent_field][value]
                else:
                    interpret_value = value
            except:
                interpret_value = "ERR interpret:" + value 
            print("{: >10.10} {: >4.4} {: <50.50} {:<3}".format(parent_field, field_type, name, interpret_value))
        except:
            print("Error parsing:", field)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data.')
    parser.add_argument('csv_file', help='CSV file to load')
    parser.add_argument('do_file', help='DO file to load')
    parser.add_argument('dct_file', help='DCT file to load')
    args = parser.parse_args()
    process(args.csv_file, args.do_file, args.dct_file)
