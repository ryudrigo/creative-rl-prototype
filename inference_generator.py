import os
import time
import gym
import mine_foo
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from difficulty_sensor import Sensor
import tensorflow as tf


def infer(sensor = None, trials=1, render=False):

    debug=False

    if render:
        from gym.envs.classic_control import rendering
        viewer = rendering.SimpleImageViewer()

    env_name = 'three-cars-pcgrl-v0'
    
    solver_path=os.path.join('solver_model','best_model')
    player_agent = PPO.load(solver_path)
    
    generator_path=os.path.join('generator_model','best_model')
    generator_agent= DQN.load(generator_path)

    env = gym.make(env_name)
    
    prediction_total_mean=0
    total_steps =0
    for i in range(trials):
        obs = env.reset()
        dones = False
        generator_counter = 0
        total_predictions=0
        predictions_counter=0        
        generator_dones=False
        while not dones:
            player_action, _states = player_agent.predict(obs)
            generator_action =generator_agent.predict(obs)[0]
            
            if render:
                viewer.imshow(env.render())
                time.sleep(0.05)
                #to record a video:
                #if total_steps ==0:
                #    time.sleep(10)
            if debug:
                print ('player_action', player_action)
                print ('generator_action', generator_action)
            obs, rewards, dones, info = env.step((player_action, generator_action))
                            
            if sensor is not None:
                total_predictions+=sensor.predict(np.expand_dims(obs, 0))
                predictions_counter+=1
                    
            total_steps+=1
            if dones:
                break
        if sensor is not None:
            prediction_mean = total_predictions/predictions_counter
            if i%10==0:
                print(prediction_mean)
            prediction_total_mean+=prediction_mean
    if sensor is not None:
        prediction_total_mean = prediction_total_mean/trials
        print ('total mean', prediction_total_mean)
        print ('difficulty per measurement', prediction_total_mean)
        
def infer_with_sensor():
    sensor = Sensor()
    sensor.load()
    infer(trials=50,sensor=sensor, render=False)

################################## MAIN ########################################

if __name__ == '__main__':    
   infer(render=True, trials=5)
   #infer_with_sensor()
