import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image
import os
import time
import random


class ThreeCars(gym.Env):
  
  def __init__(self):
    self.SPACE_H=6
    self.SPACE_W=3
    self.images_path = 'mine-foo/mine_foo/envs/images/'
    self.seed()
    self.observation_space = spaces.Box(
        low=0, high=3, shape=(self.SPACE_H, self.SPACE_W), dtype= np.uint8
    )
    self.action_space = spaces.Discrete(3)
    
    self.state = np.array([[0,0, 0],[0,0, 0],[0,0, 0],[0,0,0],[0,0, 0], [2,3, 2]])
    self.reward = 1
    self.lives = 1
    self.dones = False
    self.image_size = 64
    self.imageMapping = {
        0:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        1:Image.open(self.images_path + 'enemy.png').convert('RGBA').resize((self.image_size, self.image_size)),
        2:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        3:Image.open(self.images_path + 'player.png').convert('RGBA').resize((self.image_size, self.image_size))
    }
        
  def step(self, action):
  
    #Convert form Discrete action space
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
        self.lives= self.lives-1
        if self.lives<=0:
            self.dones=True
    if self.dones==False:
        #spawn car
        position = random.randint(0, 2)
        willappear = random.randint(0, 9)
        new_spawn = np.array([[0, 0, 0]])
        if willappear <5:
            new_spawn[0][position] = 1    
        self.state[:-1] = np.concatenate((new_spawn,self.state[0:-2]))
    
    return self.state, reward, self.dones, {}
    
    
  def reset(self):
    self.seed()
    self.lives=1
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
