import time
import gym
import mine_foo
import numpy as np
from stable_baselines3 import PPO
from difficulty_sensor import Sensor
import tensorflow as tf
import os

def infer(sensor = None, render = False, trials =1):

    if render:
        from gym.envs.classic_control import rendering
        viewer = rendering.SimpleImageViewer()
        
    model_path=os.path.join('solver_model','best_model')
    env_name = 'three-cars-hard-v0'
    
    tf.compat.v1.disable_eager_execution()
    agent = PPO.load(model_path)
    
    env = gym.make(env_name)
    
    total_rewards=0
    total_levels=None
    total_deaths=None
    prediction_total_mean=0
    predictions_counter_accumulator=0
    total_steps=0
    for i in range(trials):
        obs = env.reset()
        dones = False
        levels = np.empty((1, 6, 3))
        deaths = []
        total_predictions=0
        total_rewards=0
        predictions_counter=0
        last_length=0
        while not dones:
            last_length+=1
            action, _states = agent.predict(obs)
            obs, rewards, dones, info = env.step(action)
  
            levels=np.concatenate((levels, np.expand_dims(obs, 0)))
            deaths.append(info['died'])
            if sensor is not None:
                total_predictions+=sensor.predict(np.reshape(obs, (1, 6, 3)))
                predictions_counter+=1
            total_rewards=total_rewards+rewards
            if render:
                viewer.imshow(env.render())
                time.sleep(0.1)
            if dones:
                deaths[-1]=True
                levels=levels[:-2]
                deaths=deaths[1:]
                break
            total_steps+=1
        if sensor is not None:
            prediction_mean = total_predictions/predictions_counter
            if i%10==0:
                print(prediction_mean)
            prediction_total_mean+=prediction_mean
        if total_deaths is None:
            total_deaths=deaths
        if total_levels is None:
            total_levels=levels

        total_deaths = np.concatenate((total_deaths, deaths))
        total_levels = np.concatenate((total_levels, levels))
    if sensor is not None:
        prediction_total_mean = prediction_total_mean/trials
        print ('total mean', prediction_total_mean)
        print ('difficulty per measurement', prediction_total_mean)
    if render:
        viewer.close()
    return total_levels, total_deaths
    

def train_sensor(levels, deaths):
    sensor = Sensor()
    sensor.train(np.reshape(levels, (-1, 6, 3)), np.array(deaths, dtype=np.int64))

def train():
    levels, deaths = infer(trials=1000)
    train_sensor(levels, deaths)

def infer_with_sensor():
    sensor = Sensor()
    sensor.load()
    infer(trials=50, sensor=sensor)
    
################################## MAIN ########################################

if __name__ == '__main__':
    infer_with_sensor()
    #train()

    #infer(render=True, trials=1)
