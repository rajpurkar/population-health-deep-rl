import pandas as pd
import numpy as np
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score

def predict(file, model='endemicity_predict'):
    df = pd.read_csv(file, low_memory=False)
    train, test = train_test_split(df, test_size = 0.2)
    y_column = 'Final result of malaria from blood smear test'
    avoid_columns = [
        y_column,
        'Presence of species: falciparum (Pf)',
        'Result of malaria rapid test'
        ]
    x_columns = [col for col in train if col not in avoid_columns]
    name_to_col_index = {}
    for (i, col) in enumerate(x_columns):
        name_to_col_index[col] = i
    x_train = train[x_columns].values
    y_train = train[y_column].values
    x_test = test[x_columns].values
    y_test = test[y_column].values

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    x = x_test
    y = y_test

    if model == 'all_no':
        y_predict = [0] * len(x)
    elif model == 'endemicity_predict':
        from sklearn import tree

        x_endem = x_train[:, name_to_col_index['Malaria endemicity']]
        x_train = pd.get_dummies(x_endem).values

        x = x[:, name_to_col_index['Malaria endemicity']]
        x = pd.get_dummies(x).values

        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x)
    elif model == 'decision_tree':
        ## todo: complete
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x)
    print(classification_report(
        y, y_predict))
    print(accuracy_score(y, y_predict))
    print(f1_score(
        y, y_predict, average='binary'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict(args.file)

