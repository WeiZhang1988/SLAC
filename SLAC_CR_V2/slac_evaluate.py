#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
from wrapped_gym_env import WrappedGymEnv
from slac_agent import SlacAgent
import matplotlib.pyplot as plt


def main():
	#simulation setting
	num_episodes = 5000
	render_animation = True
	env_name = "CarRacing-v1"
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make(env_name),1)	
	slacAgent = SlacAgent()
	#----------------------------------------------------------------
	best_score = env.reward_range[0]
	score_history = [] #collections.deque(maxlen=100)
	#----------------------------------------------------------------
	slacAgent.load_models()
	#----------------------------------------------------------------
	avg_scores = []
	try:
		for i in range(num_episodes):
			observation = env.reset()
			done = False
			score = 0
			while not done:
				#action = np.random.uniform(-1,1,env.action_space.shape)
				action = slacAgent.choose_action(observation)
				next_observation, reward, step_type, done, info = env.step(action)
				if render_animation:
					env.render()
				observation = next_observation
				score += reward[0]
				done = done[0]
			score_history.append(score)
			avg_score = np.mean(score_history[-100:])
			avg_scores.append(avg_score)
			#------------------------------------------------------------
			print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
	#----------------------------------------------------------------
	finally:
		pass
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
