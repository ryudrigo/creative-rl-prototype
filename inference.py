import time
import gym
import mine_foo
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from difficulty_sensor import Sensor
import tensorflow as tf

def infer(model_path, environment_version, sensor = None, **kwargs):
    env_name = 'three-cars-hard-v{}'.format(environment_version)
    tf.compat.v1.disable_eager_execution()
    agent = PPO.load('best_model')
    
    env = gym.make(env_name)
    
    total_rewards=0
    total_levels=None
    total_deaths=None
    prediction_total_mean=0
    predictions_counter_accumulator=0
    total_steps=0
    for i in range(kwargs.get('trials', 1)):
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
            if (kwargs.get('render', False)):
                env.render()
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
        prediction_total_mean = prediction_total_mean/kwargs.get('trials', 1)
        print ('total mean', prediction_total_mean)
        print ('difficulty per measurement', prediction_total_mean)
    return total_levels, total_deaths
    

def train_sensor(levels, deaths):
    sensor = Sensor()
    sensor.train(np.reshape(levels, (-1, 6, 3)), np.array(deaths, dtype=np.int64))

def train(model_path, environment_version, **kwargs):
    kwargs['render']=False
    kwargs['trials']=1000
    levels, deaths = infer(model_path, environment_version, **kwargs)
    train_sensor(levels, deaths)

def infer_with_sensor(model_path, environment_version, **kwargs):
    kwargs['trials']= 100
    kwargs['render']=False
    sensor = Sensor()
    sensor.load()
    infer(model_path, environment_version, sensor, **kwargs)
    
################################## MAIN ########################################
model_path = 'currently-unused'
environment_version = 0
kwargs = {
    'trials': 1,
    'render':False
}

if __name__ == '__main__':
    infer_with_sensor(model_path, environment_version, **kwargs)
    #train(model_path, environment_version, **kwargs)

    #infer(model_path, environment_version, **kwargs)
