from __future__ import print_function
import sys
from test_env import EnvTest
from dataset import Dataset
from survey_env import SurveyEnv
import numpy as np

class SimulatedSurveyEnv(SurveyEnv):
    def __init__(self, config, sampler, log_file=sys.stdout):
        assert isinstance(sampler, SimulatedDataset)
        self.past_actions = []
        super(SimulatedSurveyEnv, self).__init__(config, sampler, log_file=log_file)

    def reset(self, split):
        done = self.sampler.update(self.past_actions)
        if done == True:
            return None
        self.past_actions = []
        return super(SimulatedSurveyEnv, self).reset(split)

    def step(self, action):
        self.past_actions.append(action)
        return super(SimulatedSurveyEnv, self).step(action)

class SimulatedDataset(Dataset):
    def __init__(self, filename):
        super(SimulatedDataset, self).__init__(filename)
        self.feature_indexes = np.zeros(shape=(len(self.feature_names)), dtype=np.int8)

    def sample(self, split=None):
        x = np.zeros(shape=(1, len(self.feature_names)))
        for i in xrange(len(self.feature_names)):
            x[0, i] = self.feature_indexes[i]
        x = self._encode_X(x, self.feature_names)[0]
        y = 0
        return x, y

    def update(self, update_features):
        update_features = update_features[::-1]
        done = False
        for idx, f in enumerate(update_features):
            if f >= len(self.feature_names):
                continue
            self.feature_indexes[f] = (self.feature_indexes[f] + 1) \
                % len(self.label_encs[f].classes_)
            if self.feature_indexes[f] != 0:
                break
            elif idx == (len(update_features) - 1):
                done = True
                break
        return done


if __name__ == '__main__':
    import sys
    d = SimulatedDataset(sys.argv[1])
    count = 0
    while True:
        count += 1
        #print(d.sample())
        done = d.update([0,2])
        if done == True:
            break
    print(count)
