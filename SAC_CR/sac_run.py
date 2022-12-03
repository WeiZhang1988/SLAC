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
	def _action(self, action):
		action = self.action_space.low + (action + 1.0) * 0.5 * (self.action_space.high - self.action_space.low)
		return np.clip(action, self.action_space.low, self.action_space.high)
	def _reverse_action(self, action):
		action = 2.0 * (action - self.action_space.low) / (self.action_space.high - self.action_space.low) - 1.0
		return np.clip(action, self.action_space.low, self.action_space.high)
	def reset(self):
		observation = self.env.reset()
		reward = 0.0
		info = None
		step_type = 0	# 0 means first step
		return observation, reward, step_type, info
	def step(self, action):
		observation, reward, done, info = self.env.step(action)
		step_type = 2 if done else 1	# 2 means last step, 1 means middle step
		return observation, reward, step_type, info
#--------------------------------------------------------------------
def main():
	#hyper parameters
	actor_learning_rate = 3e-4
	critic_learning_rate = 3e-4
	alpha_learning_rate = 3e-4
	num_episodes = 500
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
		observation, reward, step_type, info = env.reset()
		done = False
		score = reward
		while not done:
			action = sacAgent.choose_action(observation)
			next_observation, next_reward, next_step_type, next_info = env.step(action)
			replay_buffer.store_transition(observation, action, reward, step_type, next_observation)
			observation = next_observation
			reward = next_reward
			step_type = next_step_type
			score += reward
			if next_step_type == 3:
				replay_buffer.store_transition(observation, action, reward, step_type, observation)
				done = True
		sacAgent.learn()
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
