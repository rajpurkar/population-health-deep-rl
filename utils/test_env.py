from __future__ import print_function
import random
import numpy as np


def sample_generate(feature_length, is_k_class, first_n):
    x = [random.choice([0, 1]) for _ in range(feature_length)]
    if is_k_class is True:
        y = [0] * (feature_length + 1)
        y[sum(x[:first_n])] = 1
    else:
        y = 1 if x[0] + x[1] == 2 else 0
    return x, y


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
    def __init__(self, shape, config=None):
        #3 states
        feature_length = shape[0]
        self.feature_length = feature_length
        self.action_space = ActionSpace(self.feature_length + 1) # extra quit action
        self.observation_space = ObservationSpace(shape)
        self.k_class = config is not None and config.k_class is True
        self.first_n = 2
        self.reset()

    def reset(self):
        self.real_state, self.y = sample_generate(self.feature_length, self.k_class, self.first_n)
        self.num_iters = 0
        self.cur_state = np.ones((self.feature_length, 1, 1)) * -1
        return self.cur_state

    def step(self, action):
        assert(action < self.feature_length + 1)
        self.num_iters += 1
        done = (self.num_iters > self.first_n) or (int(action) == self.feature_length)
        if done is True:
            if np.all(self.cur_state[:self.first_n] >= 0):
                self.reward = 10.
            else:
                self.reward = -1.
        else:
            self.reward = -2.
            self.cur_state[action] = self.real_state[action]
        
        return self.cur_state, self.y, self.reward, done, {'ale.lives':0}

    def render(self):
        print(self.cur_state)
