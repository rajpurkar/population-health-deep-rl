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
    df = pd.read_csv(file, low_memory=False)
    y_column_name = 'Final result of malaria from blood smear test'
    cols = list(df.columns)
    cols.remove(y_column_name)
    X, feature_names = get_X_cols(df, cols, **params)
    y = get_Y_col(df, y_column_name)
    return X, y, feature_names

def split_data(train_frac, input_X, input_y, input_weights=None):
    total_data = len(input_y)
    print(total_data)
    num_train = int(train_frac * total_data)
    assert num_train > 0

    train_X = input_X[:num_train]
    train_y = input_y[:num_train]

    test_X = input_X[num_train:]
    test_y = input_y[num_train:]

    if input_weights is not None:
        train_weights = input_weights[:num_train]
        test_weights = input_weights[num_train:]
        return train_X, train_y, train_weights, test_X, test_y, test_weights
    else:
        return train_X, train_y, test_X, test_y

def predict_from_data(file):
    import sklearn.linear_model
    X, y, feature_names = get_X_Y_from_data(file, flatten=True)
    clf = sklearn.linear_model.LogisticRegression()
    predictions = cross_val_predict(clf, X, y, n_jobs=-1, verbose=1)
    score = sklearn.metrics.precision_recall_fscore_support(y, predictions, average='binary')
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict_from_data(args.file)
