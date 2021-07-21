import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image
import os
import time
import random


class ThreeCarsHard(gym.Env):

  
  def __init__(self):
    self.SPACE_H=6
    self.SPACE_W=3
    self.images_path = 'mine-foo/mine_foo/envs/images/'
    self.seed()
    self.observation_space = spaces.Box(
        low=0, high=3, shape=(self.SPACE_H, self.SPACE_W), dtype= np.uint8
    )
    self.action_space = spaces.Discrete(3)
    
    self.spawnBuffer=np.delete(np.array([[0,0,0]]), -1, 0)
    self.state = np.array([[0,0, 0],[0,0, 0],[0,0, 0],[0,0,0],[0,0, 0], [2,3, 2]])
    self.reward = 1
    self.lives = 4
    self.should_get_harder=True
    #sensor training values
    #self.double_spawn_chance=20
    #self.single_spawn_chance=60

    #current values
    self.double_spawn_chance=10
    self.single_spawn_chance=40
    
    self.died=False
    self.dones = False
    self.image_size = 64
    self.imageMapping = {
        0:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        1:Image.open(self.images_path + 'enemy.png').convert('RGBA').resize((self.image_size, self.image_size)),
        2:Image.open(self.images_path + 'empty.png').convert('RGBA').resize((self.image_size, self.image_size)),
        3:Image.open(self.images_path + 'player.png').convert('RGBA').resize((self.image_size, self.image_size))
    }
  def make_new_spawn_line(self):
    spawn_chance = random.randint(0,100)
    if spawn_chance<self.double_spawn_chance:
        position_free = random.randint(0,2)
        line = np.array([[1, 1, 1]])
        line[0][position_free] = 0
    elif spawn_chance<self.single_spawn_chance:
        position_occupied = random.randint(0,2)
        line = np.array([[0,0,0]])
        line[0][position_occupied]=1
    else:
        line = np.array([[0,0,0]])
    return line
    
  def get_new_spawn(self):
    if len(self.spawnBuffer)==0:
        self.spawnBuffer = self.make_new_spawn_line()
        #create new spawn buffer
        for i in range (self.SPACE_H-1): #-1 was because we already made one line
            self.spawnBuffer=np.concatenate((self.spawnBuffer, self.make_new_spawn_line()))
    #get last line of buffer
    last_row = self.spawnBuffer[-1]
    new_spawn = np.reshape(last_row, (1, len(last_row)))
    
    #remove that line from buffer
    self.spawnBuffer = np.delete(self.spawnBuffer, -1, 0)
    return new_spawn
        
  def get_info(self):
    return {'died':self.died}
  
  def get_harder(self):
    return
  
  def step(self, action):
  
    self.died=False
    self.get_harder()
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
        self.died=True
        self.lives= self.lives-1
        if self.lives<=0:
            self.dones=True
    if self.dones==False:
        #spawn car
        new_spawn = self.get_new_spawn()            
        self.state[:-1] = np.concatenate((new_spawn,self.state[0:-2]))
    
    return self.state, reward, self.dones, self.get_info()
    
    
  def reset(self):
    self.seed()
    self.lives=4
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
