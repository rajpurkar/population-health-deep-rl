from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, filename):
        self.label_encs = {}
        self._load(filename)


    def _load(self, filename):
        df = pd.read_csv(filename, low_memory=False)
        self.feature_names = list(df.columns)
        self.y_column_name = 'Final result of malaria from blood smear test'
        self.feature_names.remove(self.y_column_name)
        X = self.process_X(df, self.feature_names)
        y = self.process_Y(df, self.y_column_name)
        self.state_shape = X[0].shape

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)


    def _sample(self, X, y):
        positive_weight = (len(y) - np.sum(y)) / (1.0 * np.sum(y))
        weights = np.zeros_like(y).astype('float')
        weights[y == 1] = positive_weight
        weights[y == 0] = 1.0
        weights = weights / np.sum(weights)

        index = np.random.choice(range(len(X)), p=weights)
        return X[index, :], y[index]


    def sample(self, split='train'):
        assert(split in ['train', 'test'])
        if split == 'train':
            return self._sample(self.X_train, self.y_train)
        else:
            return self._sample(self.X_test, self.y_test)


    def process_X(self, df, cols):
        X = df.loc[:, cols]
        max_classes = 0
        self.label_encs = []
        for col in X:
            label_enc = LabelEncoder()
            X.loc[:, col] = label_enc.fit_transform(X.loc[:, col])
            max_classes = max(len(label_enc.classes_), max_classes)
            self.label_encs.append(label_enc)
        enc = OneHotEncoder(n_values=max_classes, sparse=False)
        X = enc.fit_transform(X.values)
        X = X.reshape((X.shape[0], len(cols), 1, -1))
        return X


    def process_Y(self, df, col_name):
        y = df.loc[:, col_name].values
        y[y == 'Negative'] = 0
        y[y == 'Positive'] = 1
        return y.astype('int')


if __name__ == '__main__':
    import sys
    d = Dataset(sys.argv[1])
    for i in range(50):
        print(d.sample()[1]) # should be ~equal 0 and 1
