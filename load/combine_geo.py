from __future__ import print_function
from dbfread import DBF
import pandas as pd
import csv
import argparse
import process
import postprocess
from tqdm import tqdm

def geo_read(dbf_file):
    dbf = DBF(dbf_file)
    frame = pd.DataFrame(iter(dbf))
    return frame
  

def survey_read(data_folder, fields_to_keep):
    (reader, 
        fields_to_names,
        fields_to_parent_fields,
        field_to_values_to_label,
        fields_to_types) = process.load_structures(data_folder)
    headers = reader.fieldnames
    df = pd.DataFrame()  
    for row in tqdm(reader):
        output = {}
        for field in headers:
            name = process.convert_field_to_name(field, fields_to_names)
            if name not in fields_to_keep:
                continue
            value = row[field].strip()
            interpret_value = process.interpret_val(
                field,
                fields_to_parent_fields,
                value,
                field_to_values_to_label)
            output[name] = interpret_value
        df = df.append(output, ignore_index=True)
    return df


def combine_geo(dbf, data_folder): 
    cluster_field = 'Cluster number'
    output_field = 'Final result of malaria from blood smear test'
    endemicity_field = 'Malaria endemicity'

    geo_df = geo_read(dbf)
    geo_df = geo_df.rename(columns={'DHSCLUST': cluster_field})
    geo_df[cluster_field] = geo_df[cluster_field].astype(int)

    survey_df = survey_read(data_folder, [
        cluster_field,
        output_field,
        endemicity_field])
    survey_df = postprocess.filter_na_for_field(survey_df, output_field)
    survey_df[cluster_field] = survey_df[cluster_field].astype(int)

    merged = survey_df.merge(geo_df, how='left', on=cluster_field)
    merged = merged.rename(columns={
        'URBAN_RURA': 'URBAN_RURAL',
        output_field: 'MALARIA',
        endemicity_field: 'ENDEMICITY'
    })

    cols_to_keep = [
        'MALARIA',
        'ENDEMICITY',
        'LONGNUM',
        'LATNUM',
        'URBAN_RURAL'
    ]

    merged.to_csv(data_folder + '/combine_geo.csv',
        columns=cols_to_keep,
        mode='w',
        index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load data.')
    parser.add_argument('dbf', help='dbf file to load')
    parser.add_argument('survey_folder', help='folder of survey')
    args = parser.parse_args()
    combine_geo(args.dbf, args.survey_folder)