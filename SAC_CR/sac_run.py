#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).

import gym
import numpy as np
from wrapped_gym_env import WrappedGymEnv, ActionObservationNormalizer
from replay_buffer import ReplayBuffer
from sac_agent import SacAgent
from utils import plot_learning_curve

def main():
	#simulation setting
	use_wrapped_env = False
	load_checkpoint = False
	activate_learning = True
	activate_pre_learning_random_game = True
	render_animation = False
	#hyper parameters
	actor_learning_rate = 3e-4
	critic_learning_rate = 3e-4
	alpha_learning_rate = 3e-4
	num_random_episodes = 10
	num_episodes = 5000
	num_learning_iter = 10
	figure_file = 'plots/sac_cr.png'
	#----------------------------------------------------------------
	if use_wrapped_env:
		env = WrappedGymEnv(gym.make("CarRacing-v1", domain_randomize=True))
	else:	
		env = gym.make("CarRacing-v1", domain_randomize=True)
		aoNormal = ActionObservationNormalizer(env.action_space, env.observation_space)
	replay_buffer = ReplayBuffer()
	sacAgent = SacAgent(replay_buffer, actor_learning_rate, \
    		critic_learning_rate, alpha_learning_rate)
	#----------------------------------------------------------------
	best_score = env.reward_range[0]
	score_history = []
	#----------------------------------------------------------------
	if load_checkpoint:
		sacAgent.load_models()
	#----------------------------------------------------------------
	if activate_pre_learning_random_game:
		for i in range(num_random_episodes):
			print("enter pre-learning game [", i, "] ...")
			if use_wrapped_env:
				observation = env.reset()
			else:
				observation = aoNormal.normalizeObservation(env.reset())
			done = False
			step = 0
			while not done:
				if use_wrapped_env:
					action = np.random.uniform(-1,1,env.action_space.shape)
					next_observation, reward, step_type, done, info = env.step(action)
				else:
					action = np.random.uniform(-1,1,env.action_space.shape)
					action = aoNormal.normalizeAction(action)
					next_observation, reward, done, info = env.step(action)
					next_observation = aoNormal.normalizeObservation(next_observation)
					step_type = 1
				if render_animation:
					env.render()
				sacAgent.replay_buffer.store_transition(observation, action, reward, step_type, done, next_observation)
				observation = next_observation
				step += 1
				if done:
					print("end after ", step, " steps")
	#----------------------------------------------------------------
	for i in range(num_episodes):
		if use_wrapped_env:
			observation = env.reset()
		else:
			observation = aoNormal.normalizeObservation(env.reset())
		done = False
		score = 0
		step = 0
		while not done:
			if use_wrapped_env:
				action = sacAgent.choose_action(observation)
				next_observation, reward, step_type, done, info = env.step(action)
			else:
				action = sacAgent.choose_action(observation)
				action = aoNormal.normalizeAction(action)
				next_observation, reward, done, info = env.step(action)
				next_observation = aoNormal.normalizeObservation(next_observation)
				step_type = 1
			if render_animation:
				env.render()
			sacAgent.replay_buffer.store_transition(observation, action, reward, step_type, done, next_observation)
			observation = next_observation
			score += reward
			step += 1
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
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
	x = [i+1 for i in range(n_games)]
	plot_learning_curve(x, score_history, figure_file)
#--------------------------------------------------------------------
if __name__ == '__main__':
  main()
