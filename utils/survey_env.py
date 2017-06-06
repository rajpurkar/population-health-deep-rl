from __future__ import print_function
import random
import numpy as np
from load.predict import *
from test_env import ActionSpace
from test_env import ObservationSpace
from test_env import EnvTest
from configs.survey_env import RewardConfig

def get_next(input_X, input_y, counter):
    return input_X[counter], input_y[counter]

class SurveyEnv(EnvTest):
    def __init__(self, survey_file, config):
        # EnvTest.__init__(self, config)
        self.input_X, self.input_y, self.feature_names = get_X_Y_from_data(survey_file)
        self.counter = 0
        self.max_episodes = self.input_X.shape[0]
        self.max_steps = config.max_steps
        self.num_classes = 2
        self.config = config
        self.reward_config = RewardConfig(self.feature_names)
        self.state_shape = self.input_X.shape[1:]
        self.feature_length = self.state_shape[0]

        assert(self.config.max_steps <= self.feature_length)

        self.action_space = ActionSpace(self.feature_length + self.num_classes) # extra quit actions
        self.observation_space = ObservationSpace(self.state_shape)

        self.reset()

    def set_reward(self, done, action):
        if done is True:
            if self.y == int(action - self.feature_length):
                self.reward = self.reward_config.correctAnswerReward
            else:
                self.reward = self.reward_config.wrongAnswerReward
        else:
            self.reward = self.reward_config.get_reward(self.feature_names[action])
            self.cur_state[action] = self.real_state[action]

    def reset(self):
        self.real_state, self.y = get_next(self.input_X, self.input_y, self.counter)
        self.counter = self.counter + 1 % self.max_episodes
        self.num_iters = 0
        self.cur_state = np.ones((self.state_shape)) * -1
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state
