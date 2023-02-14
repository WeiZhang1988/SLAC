import numpy as np
import time
import gym
import carla_rl_env
import tracemalloc
import gc

params ={
'carla_port': 2000,
'map_name': 'Town01',
'window_resolution': [1620,1080],
'grid_size': [3,3],
'sync': True,
'no_render': False,
'display_sensor': True,
'ego_filter': 'vehicle.dodge.charger_police_2020',
'num_vehicles': 20,
'num_pedestrians': 20,
'enable_route_planner': True, 
}

env = gym.make('CarlaRlEnv-v0', params=params)
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
        env.display()
        action=(np.array([0.8,0.0,0.0]),np.array([False]))
        obs, rwd, done, info = env.step(action)
    gc.collect()
    print("memory usage",tracemalloc.get_traced_memory())
env.close()
tracemalloc.stop()
