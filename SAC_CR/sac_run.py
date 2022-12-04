#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
from replay_buffer import ReplayBuffer
from sac_agent import SacAgent
from utils import plot_learning_curve

class WrappedGymEnv(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		self.num_step = 0
	def _action(self, action):
		action = self.action_space.low + (action + 1.0) * 0.5 * (self.action_space.high - self.action_space.low)
		return np.clip(action, self.action_space.low, self.action_space.high)
	def _reverse_action(self, action):
		action = 2.0 * (action - self.action_space.low) / (self.action_space.high - self.action_space.low) - 1.0
		return np.clip(action, self.action_space.low, self.action_space.high)
	def reset(self):
		self.num_step = 0
		return self.env.reset()
	def step(self, action):
		action = self._action(action)
		observation, reward, done, info = self.env.step(action)
		# 2 means last step, 1 means middle step, 0 means first step
		if done:
			step_type = 2
		elif self.num_step>0:
			step_type = 1
		else:
			step_type = 0
		self.num_step += 1
		return observation, reward, step_type, info
#--------------------------------------------------------------------
def main():
	#hyper parameters
	actor_learning_rate = 3e-4
	critic_learning_rate = 3e-4
	alpha_learning_rate = 3e-4
	num_episodes = 500
	num_learning_iter = 3
	figure_file = 'plots/sac_cr.png'
	#----------------------------------------------------------------
	env = WrappedGymEnv(gym.make("CarRacing-v1", domain_randomize=True))
	replay_buffer = ReplayBuffer()
	sacAgent = SacAgent(replay_buffer, actor_learning_rate, \
    		critic_learning_rate, alpha_learning_rate)
	#----------------------------------------------------------------
	best_score = env.reward_range[0]
	score_history = []
	load_checkpoint = False
	#----------------------------------------------------------------
	for i in range(num_episodes):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = sacAgent.choose_action(observation)
			next_observation, reward, step_type, next_info = env.step(action)
			replay_buffer.store_transition(observation, action, reward, step_type, next_observation)
			observation = next_observation
			score += reward
			if step_type == 2:
				done = True
			sacAgent.learn()
		#for _ in range(num_learning_iter):	
		#------------------------------------------------------------
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
		if not load_checkpoint:
			sacAgent.save_models()
		print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
	#----------------------------------------------------------------
	if not load_checkpoint:
		x = [i+1 for i in range(n_games)]
		plot_learning_curve(x, score_history, figure_file)
    
if __name__ == '__main__':
  main()
