from __future__ import print_function
from test_env import EnvTest


class SurveyEnv(EnvTest):
    def __init__(self, config, sampler):
        config.state_shape = sampler.state_shape
        col_to_reward = {}
        for name in config.queryRewardMap:
            col_to_reward[sampler.name_to_col(name)] = \
                config.queryRewardMap[name]
        config.queryRewardMap = col_to_reward 
        super(SurveyEnv, self).__init__(config, sampler)