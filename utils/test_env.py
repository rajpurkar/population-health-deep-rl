from __future__ import print_function
import random
import numpy as np
from pprint import PrettyPrinter
pp = PrettyPrinter(depth=6)

class SampleDataset(object):
    def __init__(self, feature_length=None, first_n=None):
        assert(first_n <= feature_length)
        self.feature_length = feature_length
        self.first_n = first_n
        
    def sample(self, split):
        assert(split in ['train', 'test'])
        x = [random.choice([0, 1]) for _ in range(self.feature_length)]
        y = sum(x[:self.first_n])
        return np.array(x), y


def sample_generate(feature_length=None, first_n=None):
    assert(first_n <= feature_length)
    x = [random.choice([0, 1]) for _ in range(feature_length)]
    y = sum(x[:first_n])
    return np.array(x), y


class ActionSpace(object):
    def __init__(self, n):
        self.n = n
        self.rem_actions = None

    def sample(self, no_sample_repeats=False):
        if no_sample_repeats is True:
             return random.choice(self.rem_actions)
        else:
            return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape


class EnvLogger(object):
    def __init__(self, sampler):
        self.steps = None
        self.sampler = sampler

    def clear(self):
        self.steps = []

    def add_step(self, action):
        self.steps.append(action)

    def render_path(self):
        path = []
        for step in self.steps:
            if step < len(self.sampler.feature_names):
                path.append(self.sampler.col_to_name(step))
            else:
                path.append("Predict " + str(step - len(self.sampler.feature_names)))
        pp.pprint(path)


class EnvTest(object):
    def __init__(self, config, sampler):
        self.logger = EnvLogger(sampler)
        self.max_steps = config.max_steps
        self.num_classes = config.num_classes
        self.config = config
        self.feature_length = config.state_shape[0]
        self.sampler = sampler

        assert(self.config.max_steps <= self.feature_length)

        self.action_space = ActionSpace(self.feature_length + self.num_classes) # extra quit actions
        self.observation_space = ObservationSpace(self.config.state_shape)

    def reset(self, split):
        assert(split in ['train', 'test'])
        self.logger.clear()
        self.mode = split
        self.real_state, self.y = self.sampler.sample(self.mode)
        self.num_iters = 0
        self.cur_state = np.ones_like(self.real_state) * -1
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state

    def step(self, action):
        assert(self.mode is not None)
        assert(action < self.feature_length + self.num_classes)
        self.logger.add_step(action)
        self.num_iters += 1

        if self.config.no_repeats is True or self.config.no_sample_repeats is True:
            if action in self.action_space.rem_actions:
                self.action_space.rem_actions.remove(action)

        done = (self.num_iters > self.max_steps) or (int(action) >= self.feature_length)
        if done is True:
            if self.y == int(action - self.feature_length):
                reward = self.config.correctAnswerReward
            else:
                reward = self.config.wrongAnswerReward
            if self.mode == 'test':
                self.logger.render_path()
        else:
            if (self.config.queryRewardMap and
                    action in self.config.queryRewardMap):
                reward = self.config.queryRewardMap[action]
            else:
                reward = self.config.queryReward
            self.cur_state[action] = self.real_state[action]
        return self.cur_state, reward, done

    def render(self):
        print(self.cur_state)


if __name__ == '__main__':
    sampler = SampleDataset(feature_length=5, first_n=2)
    for i in range(20):
        sample = sampler.sample('train')
        print(sample)
