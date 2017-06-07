from __future__ import print_function
from test_env import EnvTest


class SurveyEnv(EnvTest):
    def __init__(self, c, sampler):
        c.state_shape = sampler.state_shape
        super(SurveyEnv, self).__init__(c, sampler)