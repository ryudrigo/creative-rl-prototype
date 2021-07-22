#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from mine_foo.envs.three_cars_pcgrl_RLlib import ThreeCarsPCGRL_RLlib


import tensorflow as tf
import numpy as np
import os
import gym
import mine_foo

n_steps = 0
log_dir = './pcgrl-train'
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
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"]= ThreeCarsPCGRL_RLlib
    config["num_workers"]= num_cpu
    config["num_gpus"]= 0
    #config["timesteps_per_iteration"]=61
    #config['log_level']='INFO'
    stop={
        'timesteps_total':steps
    }
    
    
    experiment_name= "PPO_PCGRL"
    experiment_name_with_suffix= "PPO_PCGRL-2021-07-21_00-20-52"
    
    # to continue from where we last stopped
    #results = tune.run('PPO',config=config, stop=stop,keep_checkpoints_num=1,checkpoint_at_end=True, checkpoint_score_attr="episode_reward_mean", metric ="episode_reward_mean", mode='max', name=experiment_name_with_suffix, resume =True)
    
    #to train anew
    results = tune.run('PPO',config=config, stop=stop,checkpoint_freq =10,checkpoint_at_end=True, checkpoint_score_attr="episode_reward_mean", metric ="episode_reward_mean", mode='max', name = experiment_name)
    ray.shutdown()
################################## MAIN ########################################
steps = 4e5
render = True
logging = True
environment_version = 0
num_cpu=1

if __name__ == '__main__':
    main(0, steps, render, num_cpu)
