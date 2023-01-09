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
    agent.load_models()
  
    n_games = 10000
  
    best_score = env.reward_range[0]
    score_history = []
  
    avg_score = 0
    
    step = 0

    try:
        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                step += 1
                action = agent.choose_action_deterministic(observation)
                next_observation,reward,step_type,done,info = env.step(action)
                env.render()
                score += reward[0]
                done = done[0]
                observation = next_observation

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
    
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
  
    finally:
        pass    

if __name__ == '__main__':
  main()


