#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import gym
import numpy as np
import collections
from wrapped_gym_env import WrappedGymEnv
from slac_agent import SlacAgent
import matplotlib.pyplot as plt

def main():
	#simulation setting
	load_checkpoint = True
	activate_learning = True
	activate_pre_learning_random_game = False
	render_animation = False
	env_name = "CarRacing-v1"
	#hyper parameters
	num_random_episodes = 50
	num_episodes = 40000
	num_learning_iter = 10
	figure_file = 'tmp/slac_cr.png'
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make(env_name),4)	
	slacAgent = SlacAgent()
	#----------------------------------------------------------------
	best_score = env.reward_range[0]
	score_history = [] #collections.deque(maxlen=100)
	#----------------------------------------------------------------
	if load_checkpoint:
		slacAgent.load_models()
	#----------------------------------------------------------------
	if activate_pre_learning_random_game:
		for i in range(num_random_episodes):
			print("enter pre-learning game ", i, " ... ")
			observation = env.reset()
			done = False
			step = 0
			while not done:
				action = np.random.uniform(-1,1,env.action_space.shape)
				next_observation, reward, step_type, done, info = env.step(action)
				if render_animation:
					env.render()
				slacAgent.store_transition(observation, action, reward, step_type, done, next_observation)
				observation = next_observation
				step += 1
	#----------------------------------------------------------------
	avg_scores = []
	try:
		for i in range(num_episodes):
			observation = env.reset()
			done = False
			score = 0
			step = 0
			while not done:
				action = slacAgent.choose_action(observation)
				next_observation, reward, step_type, done, info = env.step(action)
				if render_animation:
					env.render()
				slacAgent.store_transition(observation, action, reward, step_type, done, next_observation)
				observation = next_observation
				score += reward[0]
				step += 1
				done = done[0]
			score_history.append(score)
			avg_score = np.mean(score_history[-100:])
			avg_scores.append(avg_score)
			#------------------------------------------------------------
			print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
			if avg_score > best_score:
				best_score = avg_score
				slacAgent.save_models()
			#------------------------------------------------------------
			if activate_learning:
				for _ in range(num_learning_iter):	
					slacAgent.learn()
	#----------------------------------------------------------------
	finally:
		plt.plot(range(len(score_history)),score_history)
		plt.plot(range(len(avg_scores)),avg_scores)
		plt.show()
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
