from __future__ import print_function
from test_env import EnvTest


class SurveyEnv(EnvTest):
    def __init__(self, config, sampler):
        config.state_shape = self.sampler.state_shape
        super(SurveyEnv, self).__init__(config, sampler)