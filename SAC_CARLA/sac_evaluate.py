#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
from wrapped_gym_env import WrappedGymEnv
from sac_agent import SacAgent
import matplotlib.pyplot as plt
import carla_rl_env


def main():
	#env params
	params ={
	'carla_port': 2000,
	'map_name': 'Town01',
	'window_resolution': [1620,1080],
	'grid_size': [3,3],
	'sync': True,
	'no_render': False,
	'display_sensor': False,
	'ego_filter': 'vehicle.dodge.charger_police_2020',
	'num_vehicles': 20,
	'num_pedestrians': 20,
	'enable_route_planner': True, 
	}
	#simulation setting
	num_episodes = 5000
	render_animation = True
	env_name = "CarlaRlEnv-v0"
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make(env_name, params=params))	
	sacAgent = SacAgent()
	#----------------------------------------------------------------
	best_score = -np.inf
	score_history = [] #collections.deque(maxlen=100)
	#----------------------------------------------------------------
	sacAgent.load_models()
	#----------------------------------------------------------------
	avg_scores = []
	try:
		for i in range(num_episodes):
			observation = env.reset()
			image = observation['front_camera']
			gnss = observation['gnss']
			target = observation['trgt_pos']
			done = False
			score = 0
			while not done:
				action = sacAgent.choose_action(image,gnss,target)
				next_observation, reward, step_type, done, info = \
				env.step(action)
				next_image = next_observation['front_camera']
				next_gnss = next_observation['gnss']
				next_target = next_observation['trgt_pos'] 
				if render_animation:
					env.render()
				observation = next_observation
				score += reward[0]
				done = done[0]
			score_history.append(score)
			avg_score = np.mean(score_history[-10:])
			avg_scores.append(avg_score)
			#------------------------------------------------------------
			print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
	#----------------------------------------------------------------
	finally:
		pass
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
