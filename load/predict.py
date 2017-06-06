from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import os
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import itertools


def get_X_cols(df, names, flatten=False):
    X = df.loc[:, names]
    feature_names = list(X.columns)
    max_classes = 0
    for col in X:
        label_enc = LabelEncoder()
        X.loc[:, col] = label_enc.fit_transform(X.loc[:, col])
        max_classes = max(len(label_enc.classes_), max_classes)

    enc = OneHotEncoder(n_values=max_classes, sparse=False)
    X = enc.fit_transform(X.values)
    if flatten is False:
        X = X.reshape((X.shape[0], len(feature_names), 1, -1))

    return X, feature_names


def get_Y_col(df, col_name):
    y = LabelEncoder().fit_transform(df.loc[:, col_name])
    return y


def get_X_Y_from_data(file, **params):
    assert(os.path.basename(file) == 'post-processed.csv')
    df = pd.read_csv(file, low_memory=False)
    y_column_name = 'Final result of malaria from blood smear test'
    cols = list(df.columns)
    cols.remove(y_column_name)
    X, feature_names = get_X_cols(df, cols, **params)
    y = get_Y_col(df, y_column_name)
    return X, y, feature_names


def predict_from_data(file):
    import sklearn.linear_model
    X, y = get_X_Y_from_data(file, flatten=True)
    clf = sklearn.linear_model.LogisticRegression()
    predictions = cross_val_predict(clf, X, y, n_jobs=-1, verbose=1)
    score = sklearn.metrics.precision_recall_fscore_support(y, predictions, average='binary')
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict_from_data(args.file)
