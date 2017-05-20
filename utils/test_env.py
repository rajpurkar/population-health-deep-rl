from __future__ import print_function
import random
import numpy as np


def sample_generate(feature_length, first_n):
    assert(first_n <= feature_length)
    x = [random.choice([0, 1]) for _ in range(feature_length)]
    y = sum(x[:first_n])
    return np.array(x), y

class ActionSpace(object):
    def __init__(self, n):
        self.n = n
        self.rem_actions = None

    def sample(self, no_repeat):
        if no_repeat is True:
            return random.choice(self.rem_actions)
        else:
            return np.random.randint(0, self.n)

class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape

class EnvTest(object):
    """
    Adapted from Igor Gitman, CMU / Karan Goel
    """
    def __init__(self, config):
        self.max_steps = config.max_steps
        self.num_classes = config.num_classes
        self.config = config
        self.feature_length = config.state_shape[0]

        assert(self.config.max_steps <= self.feature_length)
        
        self.action_space = ActionSpace(self.feature_length + self.num_classes) # extra quit actions
        self.observation_space = ObservationSpace(self.config.state_shape)

        self.reset()

    def reset(self):
        self.real_state, self.y = sample_generate(
            self.feature_length, self.max_steps) # will be replaced by real sampler
        self.num_iters = 0
        self.cur_state = np.ones((self.feature_length, 1, 1)) * -1
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state

    def step(self, action):
        assert(action < self.feature_length + self.num_classes)
        self.num_iters += 1
        if action in self.action_space.rem_actions:
            self.action_space.rem_actions.remove(action)
        done = (self.num_iters > self.max_steps) or (int(action) >= self.feature_length)
        if done is True:
            if self.y == int(action - self.feature_length):
                self.reward = self.config.correctAnswerReward
            else:
                self.reward = self.config.wrongAnswerReward
        else:
            self.reward = self.config.queryReward
            self.cur_state[action] = self.real_state[action]
        
        return self.cur_state, self.reward, done

    def render(self):
        print(self.cur_state)

if __name__ == '__main__':
    for i in range(20):
        print(sample_generate(5, 2))
