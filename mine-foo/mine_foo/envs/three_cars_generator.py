import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image
import os
import time
import random
from difficulty_sensor import Sensor
#from stable_baselines3 import PPO
from stable_baselines3 import PPO

class ThreeCarsGenerator(gym.Env):

  def __init__(self):
    self.generator_training=True
    self.SPACE_H=6
    self.SPACE_W=3
    self.steps_limit=99
    self.moving_window_data=[]
    self.generator_reward_window_size=1
    self.images_path = 'mine-foo/mine_foo/envs/images/'
    solver_path=os.path.join('solver_model','best_model')
    self.sensor = Sensor()
    self.debug=False
    self.sensor.load()
    self.seed()
    self.observation_space = spaces.Box(
        low=0, high=3, shape=(self.SPACE_H, self.SPACE_W), dtype= np.int64
    )
    if self.generator_training:
        self.action_space = spaces.Discrete(8)
    else:
        self.action_space =spaces.Tuple((spaces.Discrete(3), spaces.Discrete(8)))
    self.solver_agent = PPO.load(solver_path)

    #first space(1) is player controls, 0=left, 1=wait, 2=right
    #second space(2) is a tuple of level generator controls.
    #each element of tuple corresponds to one of the 3 positions of new line
    
    self.state = np.array([[0,0, 0],[0,0, 0],[0,0, 0],[0,0,0],[0,0, 0], [2,3, 2]])
    self.reward = 1 # for the player
    self.total_steps=0
    self.lives = 4
    self.intended_difficulty=0.25
    #self.intended_difficulty=0.010
    #self.intended_difficulty=0.005
    #self.intended_difficulty=0.001
    self.died=False
    self.dones = False
    self.image_size = 64
    self.imageMapping = {
        0:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        1:Image.open(self.images_path + 'enemy.png').convert('RGBA').resize((self.image_size, self.image_size)),
        2:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        3:Image.open(self.images_path + 'player.png').convert('RGBA').resize((self.image_size, self.image_size))
    }
        
  def get_new_spawn(self, generator_action):
    new_spawn = np.array([[0,1, 0]])
    return new_spawn
        
  def get_info(self):
    if self.generator_training:
        return {}
    else:
        return {'row':self.state[-4], 'died':self.died}

  def generator_step(self, action):

    #update state with generator action
    #new_spawn = np.expand_dims(action, 0)
    if action ==0:
        new_spawn= np.array([[0,0,0]])
    if action ==1:
        new_spawn= np.array([[0,0,1]])
    if action ==2:
        new_spawn= np.array([[0,1,0]])
    if action ==3:
        new_spawn= np.array([[0,1,1]])
    if action ==4:
        new_spawn= np.array([[1,0,0]])
    if action ==5:
        new_spawn= np.array([[1,0,1]])
    if action ==6:
        new_spawn= np.array([[1,1,0]])
    if action ==7:
        new_spawn= np.array([[1,1,1]])
            
    self.state[:-1] = np.concatenate((new_spawn,self.state[0:-2]))
    
    predicted_difficulty = self.sensor.predict(np.reshape(self.state, (1, 6, 3)))
    
    self.moving_window_data.append(float(np.asscalar(predicted_difficulty)))
    
    if len(self.moving_window_data)>self.generator_reward_window_size:
        self.moving_window_data.pop(0)
    average_window_data = sum(self.moving_window_data)/len(self.moving_window_data)
    reward_diffence = (self.intended_difficulty  - average_window_data)
    reward = -abs(reward_diffence)
    

    if self.debug: # and self.total_steps > 200:
        print (self.total_steps)
        print(self.moving_window_data)
        print (predicted_difficulty)
        print (sum(self.moving_window_data))
        print(len(self.moving_window_data))
        print(average_window_data)
        print (reward_diffence)
        print (reward)
    return self.state, reward, self.dones, self.get_info()
        
  def player_step(self, action):
    self.died=False
    
    #Convert from Discrete action space
    action = action-1
    
    #move player
    playerposition=0
    if self.state[-1][0]==3:
        playerposition=-1
    elif self.state[-1][1]==3:
        playerposition=0
    elif self.state[-1][2]==3:
        playerposition=1

    
    #check bounds and update state
    playerposition = playerposition + action
       
    if playerposition >=1:
        playerposition=1
        self.state[-1] = np.array([2,2,3])
    elif playerposition <=-1:
        playerposition=-1
        self.state[-1] = np.array([3,2,2])
    elif playerposition == 0:
        self.state[-1] = np.array([2,3,2])
      
    reward = self.reward
    
    #check if hits a a car
    if self.generator_training==False:
        playerposition_index = playerposition+1
        if self.state[-2][playerposition_index]==1:
            self.died=True
            self.lives= self.lives-1
            if self.lives<=0:
                self.dones=True       
    
    return self.state, reward, self.dones, self.get_info()

  def step(self, action):
    self.total_steps+=1
    if self.generator_training:
        if self.total_steps%self.steps_limit==0:
            self.dones=True
        generator_action=action
        player_action, _ = self.solver_agent.predict(self.state)
        self.player_step(player_action)
        return_data = self.generator_step(generator_action)
    else:
        player_action, generator_action = action
        return_data = self.player_step(player_action)
        self.generator_step(generator_action)
    
    if self.debug and self.total_steps%6==0 :#and self.total_steps > 200:
        print (generator_action)
        print (self.state)
        time.sleep(1)
        print (return_data[1])
    return return_data

  def reset(self):
    self.seed()
    self.lives=4
    #self.total_steps
    self.moving_window_data=[]
    self.died=False
    self.state = np.array([[0,0, 0],[0,0, 0],[0,0, 0],[0,0,0],[0,0, 0], [2,3, 2]])
    self.dones = False
    return self.state

  def convertToImage(self, matrix):
    img_list = [[self.imageMapping[element] for element in row] for row in matrix]
    img = Image.new ('RGB', (self.SPACE_W*self.image_size, self.SPACE_H* self.image_size))
    for i in range(self.SPACE_H):
        for j in range(self.SPACE_W):
            img.paste(img_list[i][j], (j*self.image_size, i*self.image_size))
    return img
  
  def render(self, mode='human', close=False):
    img = self.convertToImage(self.state)
    img = np.array(img)
    return img
  
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
