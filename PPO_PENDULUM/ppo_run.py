#!/usr/bin/env python

# Copyright (c) 2022: Wei ZHANG (wei_zhang_1988_outlook.com).
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
from wrapped_gym_env import WrappedGymEnv
import time
import numpy as np
from ppo_agent import Agent
from utils import plot_learning_curve

def main():
# Set gym-carla environment
    env = WrappedGymEnv(gym.make('Pendulum-v1'))	
    agent = Agent()
  
    n_games = 20000
    figure_file = 'plots/ppo_pendulum_entropy.png'
  
    best_score = env.reward_range[0]
    score_history = []
  
    avg_score = 0
    
    step = 0
    N = 256
    try:
        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                step += 1
                action, log_prob = agent.choose_action(observation)
                next_observation,reward,step_type,done,info = env.step(action)
                score += reward[0]
                agent.store_transition(observation, action, log_prob, reward, step_type, done, next_observation)
                done = done[0]
                observation = next_observation
                if step % N == 0:
           	        agent.learn()
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
    
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()
    
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
  
    finally:
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)      

if __name__ == '__main__':
  main()


