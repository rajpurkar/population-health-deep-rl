from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
import sklearn.linear_model
import sklearn.neural_network
import itertools
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(depth=6)

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

def get_X_cols(df, names):
    X = df[names]
    X = pd.get_dummies(X)
    feature_names = list(X.columns)
    X = X.values
    return X, feature_names


def get_Y_col(df, col_name):
    y = df[col_name]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return y

def predict(file, model='endemicity_predict'):
    df = pd.read_csv(file, low_memory=False)
    y_column_name = 'Final result of malaria from blood smear test'
    columns = list(df.columns)
    for num_features in range(len(columns), len(columns) + 1):
        print(num_features, "features")
        for cols in tqdm(findsubsets(columns, num_features)):
            cols = list(cols)
            ignore_phrase_columns = [y_column_name.lower(), 'presence of species:', 'rapid test', 'number']
            cols = filter(lambda col: not any(phrase.lower() in col.lower() for phrase in ignore_phrase_columns), cols)
            X, feature_names = get_X_cols(df, cols)
            y = get_Y_col(df, y_column_name)
            clf = sklearn.neural_network.MLPClassifier()
            #clf = sklearn.linear_model.LogisticRegressionCV()
            #clf = sklearn.linear_model.Lasso()
            predictions = cross_val_predict(clf, X, y, n_jobs=-1, verbose=1)
            score = sklearn.metrics.precision_recall_fscore_support(y, predictions, average='binary')
            #score = sklearn.metrics.f1_score(y, predictions, average='binary')
            pp.pprint(cols)
            pp.pprint(score)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict(args.file)

