from __future__ import print_function
import random
import numpy as np
from pprint import PrettyPrinter
import sys

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
    def __init__(self, feature_length, num_classes):
        self.feature_length = feature_length
        self.num_classes = num_classes
        self.n = feature_length + num_classes
        self.rem_actions = None

    def sample(self, no_sample_repeats=False, force_pred=False):
        if force_pred is True:
            return np.random.randint(self.feature_length, self.num_classes)
        if no_sample_repeats is True:
             return random.choice(self.rem_actions)
        else:
            return np.random.randint(0, self.n)


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape


class EnvLogger(object):
    def __init__(self, sampler, log_file=sys.stdout):
        self.steps = None
        self.sampler = sampler
        self.log_file = log_file

    def clear(self):
        self.steps = []

    def add_step(self, action, reward, one_hot_value):
        self.steps.append((action, reward, one_hot_value))

    def render_path(self):
        path = []
        for (action, reward, one_hot_value) in self.steps:
            if one_hot_value is not None:
                value = np.argmax(one_hot_value)
                value = self.sampler.col_value_to_interpretation(action, value)
            else:
                value = None
            if action < len(self.sampler.feature_names):
                action = self.sampler.col_to_name(action)
            else:
                path.append("Predict " + str(action - len(self.sampler.feature_names)))
                pred = str(action - len(self.sampler.feature_names))
                action = "Predict " + pred
            path.append((action, reward, value))
        self.log_file.write(pp.pformat('-------------')+"\n")
        self.log_file.write(pp.pformat(path)+"\n")
        self.log_file.write(pp.pformat('-------------\n')+"\n")


class EnvTest(object):
    def __init__(self, config, sampler, log_file=sys.stdout):
        self.logger = EnvLogger(sampler, log_file)
        self.max_steps = config.max_steps
        self.num_classes = config.num_classes
        self.config = config
        self.feature_length = config.state_shape[0]
        self.sampler = sampler

        assert(self.config.max_steps <= self.feature_length)

        self.action_space = ActionSpace(self.feature_length, self.num_classes) # extra quit actions
        self.observation_space = ObservationSpace(self.config.state_shape)

    def reset(self, split):
        assert(split in ['train', 'test'])
        self.logger.clear()
        self.mode = split
        self.real_state, self.y = self.sampler.sample(self.mode)
        self.num_iters = 0
        self.cur_state = np.zeros_like(self.real_state)
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state

    def step(self, action):
        assert(self.mode is not None)
        assert(action < self.feature_length + self.num_classes)

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
            self.logger.add_step(action, reward, None)
            if self.mode == 'test':
                self.logger.render_path()
        else:
            if (self.config.queryRewardMap and
                    action in self.config.queryRewardMap):
                reward = self.config.queryRewardMap[action]
            else:
                reward = self.config.queryReward
            self.cur_state[action, :, :] = self.real_state[action, :, :]
            self.logger.add_step(action, reward, self.real_state[action, :, :])

        return self.cur_state, reward, done

    def render(self):
        print(self.cur_state)


if __name__ == '__main__':
    sampler = SampleDataset(feature_length=5, first_n=2)
    for i in range(20):
        sample = sampler.sample('train')
        print(sample)
