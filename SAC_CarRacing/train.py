import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from agent import Agent
from replay_buffer import ReplayBuffer
from model import CarWrapper
import os


# ref:  (https://github.com/xtma/pytorch_car_caring).

continue_train = True
use_cpu = False
pre_exploration = False

if use_cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def exploration(env,replay_buffer):
    episode = 0
    while True:
        episode += 1
        observation,_ = env.reset()
        steps = 0
        while True:
            action = env.sample_action()
            next_observation, reward, done, truncated, _ = env.step(action)
            state = [observation,action,reward,next_observation,done]
            replay_buffer.append(state)
            observation = next_observation
            steps += 1
            if done or truncated:
                break
            if len(replay_buffer) > 256:
                return
        print(f'explorating: episode {episode}, buffer length {len(replay_buffer)} / {replay_buffer.length}') 
try:
    env = CarWrapper(gym.make('CarRacing-v2'))#,render_mode='human'))# ))#
    action_dim=env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    if continue_train:
        print('\n -----Continue Train----- \n')
        agent.load_models()
    replay_buffer = ReplayBuffer(5000)
    if pre_exploration:
        print('\n -----Exploration----- \n')
        exploration(env,replay_buffer)
    start_time = time.time()
    batch_size = 128
    score_list = []
    average_score_list = []
    total_eposide = 10000
    best_score = -100
    for episide in range(total_eposide):
        steps = 0
        score = 0
        observation,_ = env.reset()
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, truncated , _ = env.step(action)
            transtion = [observation,action,reward,next_observation,done]
            replay_buffer.append(transtion)
            observation = next_observation
            score += reward
            steps += 1
            if len(replay_buffer) > batch_size:
                agent.learn(replay_buffer.sample(batch_size))
            if done or truncated: # or score <0
                print(f"done {done} truncted {truncated} steps {steps}")
                break
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        t = int(time.time()-start_time)
        print(f'episode: {episide}, score: {score:.2f} ,avg_score:{average_score:.2f}, time: {t//3600:2}:{t%3600//60:2}:{t%60:2}') 
        if average_score > best_score:
            best_score = average_score
            agent.save_models()
finally:
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()
