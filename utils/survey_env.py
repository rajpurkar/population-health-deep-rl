from __future__ import print_function
import random
import numpy as np
from load.predict import *
from test_env import ActionSpace
from test_env import ObservationSpace
from test_env import EnvTest

def get_survey_data(file):
    X, y = get_X_Y_from_data(file)
    return X, y

def get_next(input_X, input_y, counter):
    return input_X[counter], input_y[counter]

class SurveyEnv(EnvTest):
    def __init__(self, surveyFile, config):
        # EnvTest.__init__(self, config)
        self.input_X, self.input_y = get_survey_data(surveyFile)
        self.counter = 0
        self.max_episodes = self.input_X.shape[0]
        self.max_steps = config.max_steps
        self.num_classes = 2
        self.config = config
        self.state_shape = self.input_X.shape[1:]
        self.feature_length = self.state_shape[0]

        assert(self.config.max_steps <= self.feature_length)

        self.action_space = ActionSpace(self.feature_length + self.num_classes) # extra quit actions
        self.observation_space = ObservationSpace(self.state_shape)

        self.reset()

    def reset(self):
        self.real_state, self.y = get_next(self.input_X, self.input_y, self.counter)
        self.counter = self.counter + 1 % self.max_episodes
        self.num_iters = 0
        self.cur_state = np.ones((self.state_shape)) * -1
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state
