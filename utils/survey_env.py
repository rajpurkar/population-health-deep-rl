from __future__ import print_function
import random
import numpy as np
from load.predict import *
from test_env import ActionSpace
from test_env import ObservationSpace
from test_env import EnvTest
from configs.survey_env import RewardConfig

def get_next(input_X, input_y, pos_ex, neg_ex):
    rand = random.random()
    threshold = float(len(neg_ex))/float(len(pos_ex)+float(len(neg_ex)))
    if rand < threshold:
        idx = random.choice(pos_ex)
    else:
        idx = random.choice(neg_ex)
    return input_X[idx], input_y[idx]

class SurveyEnv(EnvTest):
    def __init__(self, input_X, input_y, feature_names, config):
        # EnvTest.__init__(self, config)
        self.input_X = input_X
        self.input_y = input_y
        self.neg_ex = [idx for idx, k in enumerate(input_y) if k == 0]
        self.pos_ex = [idx for idx, k in enumerate(input_y) if k == 1]
        self.feature_names = feature_names
        #self.counter = 0
        self.max_episodes = self.input_X.shape[0]
        self.max_steps = config.max_steps
        self.num_classes = config.num_classes
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
        self.real_state, self.y = get_next(self.input_X, self.input_y, self.pos_ex, self.neg_ex)
        #self.counter = (self.counter + 1) % self.max_episodes
        self.num_iters = 0
        self.cur_state = np.ones((self.state_shape)) * -1
        self.action_space.rem_actions = range(self.action_space.n)
        return self.cur_state
