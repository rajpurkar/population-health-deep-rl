from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn.linear_model
import warnings
warnings.filterwarnings('ignore', 'numpy equal will not check object identity in the future')
import itertools
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(depth=6)


def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def get_X_cols(df, names, flatten=True):
    X = df[names]
    feature_names = list(X.columns)
    max_classes = 0
    for col in X:
        label_enc = LabelEncoder()
        X[col] = label_enc.fit_transform(X[col])
        max_classes = max(len(label_enc.classes_), max_classes)

    enc = OneHotEncoder(n_values=max_classes, sparse=False)
    X = enc.fit_transform(X)
    if flatten is False:
        X = X.reshape((X.shape[0], len(feature_names), 1, -1))

    return X, feature_names


def get_Y_col(df, col_name):
    y = df[col_name]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return y

def get_X_Y_from_data(file):
    df = pd.read_csv(file, low_memory=False)
    y_column_name = 'Final result of malaria from blood smear test'
    cols = list(df.columns)
    ignore_phrase_columns = [y_column_name.lower(), 'presence of species:', 'rapid test', 'number']
    cols = filter(lambda col: not any(phrase.lower() in col.lower() for phrase in ignore_phrase_columns), cols)
    X, feature_names = get_X_cols(df, cols)
    y = get_Y_col(df, y_column_name)
    return X, y


def predict_from_data(file):
    X, y = get_X_Y_from_data(file)
    clf = sklearn.linear_model.LogisticRegression()
    predictions = cross_val_predict(clf, X, y, n_jobs=-1, verbose=1)
    score = sklearn.metrics.precision_recall_fscore_support(y, predictions, average='binary')
    pp.pprint(cols)
    pp.pprint(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict_from_data(args.file)

