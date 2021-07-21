import time
import gym
import mine_foo
import numpy as np
from stable_baselines3 import PPO
from difficulty_sensor import Sensor
import ray
import ray.rllib.agents.ppo as ppo
from mine_foo.envs.three_cars_pcgrl_RLlib import ThreeCarsPCGRL_RLlib
import tensorflow as tf
import pickle


def infer(model_path, environment_version, sensor = None, **kwargs):
    env_name = 'three-cars-pcgrl-v{}'.format(environment_version)
    
    player_agent = PPO.load('best_model')
    
    
    tf.compat.v1.disable_eager_execution()
    ray.init()
    ppo_config = ppo.DEFAULT_CONFIG.copy()
    ppo_config["env"]= ThreeCarsPCGRL_RLlib
    ppo_config["num_workers"]= 1
    ppo_config["num_gpus"]= 0
    generator_agent = ppo.PPOTrainer(ppo_config)
    
  
    #intended difficulty = 0.050
    generator_agent.restore("/home/rvcam/ray_results/PPO_PCGRL/PPO_ThreeCarsPCGRL_RLlib_94d26_00000_0_2021-07-21_20-38-18/checkpoint_000050/checkpoint-50")
    
    
    
    env = gym.make(env_name)
    
    prediction_total_mean=0
    total_steps =0
    for i in range(kwargs.get('trials', 1)):
        obs = env.reset()
        dones = False
        generator_counter = 0
        total_predictions=0
        predictions_counter=0
        generator_number_of_actions=15
        generator_dones=False
        while not dones:
            
            #if generator_counter>generator_number_of_actions:
            if generator_dones == True:
                #time for the player to act
                player_action, _states = player_agent.predict(obs)
                generator_counter=0
                generator_dones=False
                generator_action=None
                if (kwargs.get('render', False)):
                    env.render()
                    time.sleep(0.2)
                    print ('player_action', player_action)
                    
                obs, rewards, dones, info = env.step((player_action, generator_action))
                                
                if sensor is not None:
                    total_predictions+=sensor.predict(np.expand_dims(obs, 0))
                    predictions_counter+=1
                    
            else:
                #time for the generator to act
                player_action=None
                _, generator_action =generator_agent.compute_action(obs)
                generator_counter+=1
                if (kwargs.get('render', False)):
                    print ('generator_action', generator_action)
                obs, rewards, dones, info = env.step((player_action, generator_action))
                generator_dones = info['generator_dones']
                
            total_steps+=1
            if dones:
                break
        if sensor is not None:
            prediction_mean = total_predictions/predictions_counter
            if i%10==0:
                print(prediction_mean)
            prediction_total_mean+=prediction_mean
    if sensor is not None:
        prediction_total_mean = prediction_total_mean/kwargs.get('trials', 1)
        print ('total mean', prediction_total_mean)
        print ('difficulty per measurement', prediction_total_mean)
        
def infer_with_sensor(model_path, environment_version, **kwargs):
    kwargs['trials']=20
    kwargs['render']=False
    sensor = Sensor()
    sensor.load()
    infer(model_path, environment_version, sensor, **kwargs)

################################## MAIN ########################################
model_path = 'currently-unused'
environment_version = 0
kwargs = {
    'trials': 1,
    'render':True
}

if __name__ == '__main__':    
   infer(model_path, environment_version, **kwargs)
   #infer_with_sensor(model_path, environment_version, **kwargs)
