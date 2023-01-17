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
env.reset()
env.step([0])
env.close()
