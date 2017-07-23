import os
import argparse

from utils.simulated_survey_env import SimulatedSurveyEnv, SimulatedDataset
from configs.simulated_env import config
from rl import ActPredDQN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to predict')
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('max_steps', type=int, help='Max number of steps')
    args = parser.parse_args()

    sampler = SimulatedDataset(args.file)

    checkpoint = args.checkpoint
    config.output_path = os.path.join(config.output_path, str(args.max_steps) + '/')

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    config.max_steps = args.max_steps
    log_file = open(os.path.join(config.output_path, "paths.txt"), 'w+')
    results_file = open(os.path.join(config.output_path, "results.txt"), 'w+')
    env = SimulatedSurveyEnv(config, sampler, log_file=log_file)
    model = ActPredDQN(env, config, results_file=results_file)
    model.run_from_restore(checkpoint)
