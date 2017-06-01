import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def predict(file, model='decision_treec'):
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

    x = x_train
    y_true = y_train
    if model == 'all_no':
        y_predict = ['Negative'] * len(x)
    elif model == 'endemicity_predict':
        x_endem = x[:, name_to_col_index['Malaria endemicity']]
        y_predict = np.array(['Negative'] * len(x))
        flips = np.random.choice([0, 1], size=(len(x),), p=[0.625, 0.375])
        y_predict[np.logical_and(x_endem == 'Lake endemic', flips == 1)] = 'Positive'
        # todo: logistic regression rather than weird random assignment
    elif model == 'decision_tree':
        ## todo: complete
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x)
    print(accuracy_score(y_true, y_predict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    args = parser.parse_args()
    predict(args.file)

