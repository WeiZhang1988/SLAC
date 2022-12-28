import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
from replay_buffer import ReplayBuffer
from wrapper import RenderWrapper,CarWrapper
import matplotlib.pyplot as plt
from agent import Agent


#env = RenderWrapper(gym.make("Ant-v4",render_mode='rgb_array',terminate_when_unhealthy=False))
env = CarWrapper(gym.make('CarRacing-v2'))
slac_agent = Agent(action_dim=3)
replay_buffer = ReplayBuffer(max_len=2000,sequence_len=8)

for eposide in range(1000):
    observation ,_ = env.reset()
    steps = 0
    score = 0
    trajectory = []
    while True:
        action = env.action_space.sample()
        #action = slac_agent.predict(observation)
        next_observation, reward, done, _ = env.step(action)
        transition = np.array([observation,action,reward,next_observation,done],dtype=object)
        trajectory.append(transition)
        steps +=1
        score += reward
        if done:
            print(f'eposide {eposide} steps {steps} score {score} done {done}')
            break
    replay_buffer.append(trajectory)
    observation_sequence_batch ,\
    action_sequence_batch ,\
    reward_sequence_batch ,\
    next_observation_sequence_batch ,\
    done_sequence_batch = replay_buffer.sample(batch_size=32)
    print('hello')

env.close()