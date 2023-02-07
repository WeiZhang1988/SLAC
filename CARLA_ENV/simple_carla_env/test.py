import numpy as np
import time
import gym
import carla_env

params ={
'carla_port': 2000,
'map_name': 'Town01',
'window_resolution': [1280,720],
'grid_size': [2,3],
'sync': True,
}

env = gym.make('CarlaEnv-v0', params=params)
obs = env.reset()
while True:
    env.display()
    action=(np.array([0.8,0.0,0.0]),np.array([False]))
    obs, r, d, inf = env.step(action)
env.close()
