from __future__ import print_function
import random
import numpy as np


def sample_generate(feature_length):
    x = [random.choice([0, 1]) for _ in range(feature_length)]
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
    def __init__(self, shape):
        #3 states
        self.reward = 1.0
        feature_length = shape[0]
        self.feature_length = feature_length
        self.action_space = ActionSpace(self.feature_length + 1) # extra quit action
        self.observation_space = ObservationSpace(shape)
        self.reset()

    def reset(self):
        self.cur_state = np.array([-1, -1, -1]).reshape(-1, 1, 1)
        self.real_state, self.y = sample_generate(self.feature_length)
        self.num_iters = 0
        return self.cur_state

    def step(self, action):
        assert(action < self.feature_length + 1)
        self.num_iters += 1
        if action < self.feature_length:
            self.cur_state[action] = self.real_state[action]
        return self.cur_state, self.y, self.reward, self.num_iters >= self.feature_length, {'ale.lives':0}

    def render(self):
        print(self.cur_state)