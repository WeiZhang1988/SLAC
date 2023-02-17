#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
import collections
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
	'display_sensor': True,
	'ego_filter': 'vehicle.dodge.charger_police_2020',
	'num_vehicles': 20,
	'num_pedestrians': 20,
	'enable_route_planner': True, 
	}
	#simulation setting
	load_checkpoint = False
	activate_learning = True
	activate_pre_learning_random_game = True
	render_animation = True
	env_name = "CarlaRlEnv-v0"
	#hyper parameters
	num_random_episodes = 10
	num_episodes = 10000
	num_learning_iter = 10
	figure_file = 'tmp/sac_carla.png'
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make(env_name, params=params))	
	sacAgent = SacAgent()
	#----------------------------------------------------------------
	best_score = -np.inf
	score_history = []
	#----------------------------------------------------------------
	if load_checkpoint:
		sacAgent.load_models()
	#----------------------------------------------------------------
	if activate_pre_learning_random_game:
		for i in range(num_random_episodes):
			print("enter pre-learning game ", i, " ... ")
			observation = env.reset()
			done = False
			step = 0
			while not done:
				action = \
				np.random.normal([1.0, -1.0, 0.0],[1.0, 1.0, 1.0],env.action_space.shape)
				next_observation, reward, step_type, done, info = \
				env.step(action)
				if render_animation:
					env.display()
				sacAgent.store_transition(observation['front_camera'], \
				observation['gnss'], observation['trgt_pos'], \
				action, reward, step_type, done, \
				next_observation['front_camera'], \
				next_observation['gnss'], \
				next_observation['trgt_pos'])
				observation = next_observation
				step += 1
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
			step = 0
			while not done:
				action = sacAgent.choose_action(image,gnss,target)
				next_observation, reward, step_type, done, info = \
				env.step(action)
				next_image = next_observation['front_camera']
				next_gnss = next_observation['gnss']
				next_target = next_observation['trgt_pos'] 
				if render_animation:
					env.display()
				sacAgent.store_transition(image, gnss, target, \
				action, reward, step_type, done, \
				next_image, next_gnss, next_target)
				observation = next_observation
				score += reward[0]
				step += 1
				done = done[0]
			score_history.append(score)
			avg_score = np.mean(score_history[-10:])
			avg_scores.append(avg_score)
			#------------------------------------------------------------
			print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
			if avg_score > best_score:
				best_score = avg_score
				sacAgent.save_models()
			#------------------------------------------------------------
			if activate_learning:
				for _ in range(num_learning_iter):	
					sacAgent.learn()
	#----------------------------------------------------------------
	finally:
		plt.plot(range(len(score_history)),score_history)
		plt.plot(range(len(avg_scores)),avg_scores)
		plt.show()
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
