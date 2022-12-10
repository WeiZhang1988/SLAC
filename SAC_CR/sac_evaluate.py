#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
from wrapped_gym_env import WrappedGymEnv
from sac_agent import SacAgent
import matplotlib.pyplot as plt


def main():
	#simulation setting
	num_episodes = 5000
	render_animation = True
	env_name = "CarRacing-v1"
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make(env_name),4)	
	sacAgent = SacAgent()
	#----------------------------------------------------------------
	best_score = env.reward_range[0]
	score_history = [] #collections.deque(maxlen=100)
	#----------------------------------------------------------------
	sacAgent.load_models()
	#----------------------------------------------------------------
	avg_scores = []
	try:
		for i in range(num_episodes):
			observation = env.reset()
			done = False
			score = 0
			while not done:
				action = sacAgent.choose_action_deterministic(observation)
				next_observation, reward, step_type, done, info = env.step(action)
				if render_animation:
					env.render()
				score += reward[0]
				done = done[0]
			score_history.append(score)
			avg_score = np.mean(score_history[-100:])
			avg_scores.append(avg_score)
			#------------------------------------------------------------
			print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
	#----------------------------------------------------------------
	finally:
		plt.plot(range(len(score_history)),score_history)
		plt.plot(range(len(avg_scores)),avg_scores)
		plt.show()
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
