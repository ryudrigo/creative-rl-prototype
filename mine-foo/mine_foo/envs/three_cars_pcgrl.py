import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image
import os
import time
import random
from difficulty_sensor import Sensor

class ThreeCarsPCGRL(gym.Env):

  def __init__(self):
    self.SPACE_H=6
    self.SPACE_W=3
    self.images_path = 'mine-foo/mine_foo/envs/images/'
    self.sensor = Sensor()
    self.sensor.load()
    self.seed()
    self.observation_space = spaces.Box(
        low=0, high=3, shape=(self.SPACE_H, self.SPACE_W), dtype= np.uint8
    )
    self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2))) ))

    #first space(1) is player controls, 0=left, 1=wait, 2=right
    #second space(2) is a tuple of level generator controls.
    #each element of tuple corresponds to one of the 3 positions of new line
    
    self.state = np.array([[0,0, 0],[0,0, 0],[0,0, 0],[0,0,0],[0,0, 0], [2,3, 2]])
    self.reward = 1 # for the player
    self.max_generator_steps=15
    self.generator_setps_count=0
    self.total_steps=0
    self.lives = 4
    self.generator_dones=False
    self.intended_difficulty=0.17
    self.is_training_env = False #FOR GENERATOR TRAINING ONLY
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
        
  def get_info(self, generator=False):
    if generator == True:
        return {'generator_dones': self.generator_dones}
    else:
        return {'row':self.state[-4], 'died':self.died}
    
  def get_generator_state(self):
    return self.state
    
  def generator_step(self, action):
    self.generator_setps_count+=1

    #update state with generator action
    cell1, cell2, cell3 = action
    new_spawn = np.array([[cell1, cell2, cell3]])
    self.state[:-1] = np.concatenate((new_spawn,self.state[0:-2]))
    generator_state = self.get_generator_state()
    if self.generator_setps_count>self.max_generator_steps:
        returned_state = self.state
    else:
        returned_state = generator_state
    predicted_difficulty = self.sensor.predict(np.reshape(self.state, (1, 6, 3)))
    returned_reward = np.asscalar(-abs(self.intended_difficulty - predicted_difficulty))
    
    
    print(self.intended_difficulty, predicted_difficulty, returned_reward)
    print(new_spawn)
    print(generator_state)
    print(generator_state[:3])
    print(self.generator_setps_count,self.max_generator_steps)
    
    if self.generator_setps_count>self.max_generator_steps:
        self.generator_setps_count=0
        self.generator_dones=True
    return returned_state, returned_reward, self.dones, self.get_info(generator=True)
        
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
    
    #check if hits a a car
    reward = self.reward
    playerposition_index = playerposition+1
    if self.state[-2][playerposition_index]==1:
        self.died=True
        self.lives= self.lives-1
        if self.lives<=0:
            self.dones=True       
    
    generator_state = self.get_generator_state()
    #the final concatenated array is just to mantain shape
    #we return generator_state because generator will be the next to look at the observation
    return generator_state, reward, self.dones, self.get_info(generator=False)

  def step(self, action):
    self.total_steps+=1
    player_action, generator_action = action
    if self.is_training_env:
        player_action=None
        if self.total_steps>99:
            self.dones=True
    if player_action is not None:
        return self.player_step(player_action)
    elif generator_action is not None:
        return self.generator_step(generator_action)

  def reset(self):
    self.seed()
    self.lives=4
    self.total_steps=0
    self.generator_setps_count=0
    self.died=False
    self.generator_dones=False
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
