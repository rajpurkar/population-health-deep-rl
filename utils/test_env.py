from __future__ import print_function
import random
import numpy as np


def sample_generate(feature_length, first_n, is_k_class):
    x = [random.choice([0, 1]) for _ in range(feature_length)]
    if is_k_class is True:
        y = np.zeros(first_n + 1)
        y[sum(x[:first_n])] = 1
    else:
        y = 1 if sum(x[:first_n]) == first_n else 0
    return np.array(x), y


class ActionSpace(object):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)

class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape

class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    """
    def __init__(self, shape, max_choices, config=None):
        self.feature_length = shape[0]
        assert(max_choices <= self.feature_length)
        self.action_space = ActionSpace(self.feature_length + 1) # extra quit action
        self.observation_space = ObservationSpace(shape)
        self.config = config
        self.max_choices = max_choices
        self.reset()

    def reset(self):
        self.k_class = self.config is not None and self.config.k_class is True
        self.real_state, self.y = sample_generate(
            self.feature_length, self.k_class, self.max_choices)
        self.num_iters = 0
        self.cur_state = np.ones((self.feature_length, 1, 1)) * -1
        return self.cur_state

    def step(self, action):
        assert(action < self.feature_length + 1)
        self.num_iters += 1
        done = (self.num_iters > self.max_choices) or (int(action) == self.feature_length)
        if done is True:
            if np.all(self.cur_state[:self.max_choices] >= 0):
                self.reward = 10.
            else:
                self.reward = -1.
        else:
            self.reward = -2.
            self.cur_state[action] = self.real_state[action]
        
        return self.cur_state, self.y, self.reward, done, {'ale.lives':0}

    def render(self):
        print(self.cur_state)

if __name__ == '__main__':
    for is_k in [False, True]:
        for i in range(20):
            print(sample_generate(4, 2, is_k))
