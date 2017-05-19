from __future__ import print_function
import random


def sample_generate(feature_length):
    x = [random.choice([0, 1]) for _ in range(feature_length)]
    y = 1 if x[0] + x[1] == 2 else 0
    return x, y


def generate_n_examples(num_examples=32, feature_length=3):
    X, Y = zip(*[sample_generate(feature_length) for _ in range(num_examples)])
    return X, Y


if __name__ == '__main__':
    print(generate_n_examples())
