from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


import tensorflow as tf
import numpy as np
import os
import gym
import mine_foo

def train_solver():
    steps=6e4
    num_cpu=1
    env_name = 'three-cars-v0'
    save_path = 'solver_model'
    log_path = '.'
    
    env = make_vec_env(env_name, n_envs=num_cpu)
    env = VecMonitor(env, log_path)
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log="./runs" )
    eval_callback = EvalCallback(env, best_model_save_path='save_path',
                             log_path=log_path, eval_freq=500,
                             deterministic=True, render=False)

    model.learn(total_timesteps=int(steps), callback=eval_callback)

################################## MAIN ########################################

if __name__ == '__main__':
    train_solver()
