import numpy as np
import time
import gym
import carla_env
import tracemalloc

params ={
'carla_port': 2000,
'map_name': 'Town01',
'window_resolution': [1280,1080],
'grid_size': [3,3],
'sync': True,
'no_render': False,
'ego_filter': 'vehicle.*',
'num_vehicles': 20,
'num_pedestrians': 20,
}

env = gym.make('CarlaEnv-v0', params=params)
done = False
obs = env.reset()
n=0
tracemalloc.start()
while True:
    n += 1
    print("run ",n," rounds")
    if done:
        obs = env.reset()
        done = False
    while not done:
        #env.display()
        action=(np.array([0.8,0.0,0.0]),np.array([False]))
        obs, rwd, done, info = env.step(action)
    print("memory usage",tracemalloc.get_traced_memory())
env.close()
tracemalloc.stop()
