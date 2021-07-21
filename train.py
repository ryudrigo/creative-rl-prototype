from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


import tensorflow as tf
import numpy as np
import os
import gym
import mine_foo

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

def make_env(env_id, rank, seed=84):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env=Monitor(env, log_dir)
        return env
    set_global_seeds(seed)
    return _init

def main(environment_version, steps, render, num_cpu):
    env_name = 'three-cars-v{}'.format(environment_version)
    env = make_vec_env(env_name, n_envs=num_cpu)
    env = VecMonitor(env, log_dir)
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./runs" )
    eval_callback = EvalCallback(env, best_model_save_path='./',
                             log_path=log_dir, eval_freq=500,
                             deterministic=True, render=False)

    model.learn(total_timesteps=int(steps), callback=eval_callback)

################################## MAIN ########################################
steps = 6e4
render = True
logging = True
environment_version = 0
num_cpu=4

if __name__ == '__main__':
    main(0, steps, render, num_cpu)
