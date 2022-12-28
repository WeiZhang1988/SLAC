import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
from replay_buffer import ReplayBuffer
from wrapper import RenderWrapper,CarWrapper
import matplotlib.pyplot as plt
from agent import Agent
import time

continue_train = False
start_time = time.time()
batch_size = 64
score_list = []
average_score_list = []
total_eposide = 10000
best_score = 350
learn_number = 10
best_alpha = None
replay_buffer = ReplayBuffer(max_len=5000,sequence_len=8)

try:
    f = open('log.txt','w')
    
    env = CarWrapper(gym.make('CarRacing-v2'))#,render_mode='human'))
    action_dim=env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(action_dim=action_dim)
    if continue_train:
        print('\n -----Continue Train----- \n')
        agent.load_models()
    for episode in range(total_eposide):
        observation ,_ = env.reset()
        steps = 0
        score = 0
        trajectory = []
        while True:
            action = agent.predict(observation)
            next_observation, reward, done, _ = env.step(action)
            transition = np.array([observation,action,reward,next_observation,done],dtype=object)
            trajectory.append(transition)
            steps +=1
            score += reward
            if done:
                break
        replay_buffer.append(trajectory)
        if len(replay_buffer) > batch_size:
            for _ in range(learn_number):
                agent.learn(replay_buffer.sample(batch_size=64))
        score_list.append(score)
        average_score = np.mean(score_list[-100:])
        average_score_list.append(average_score)
        log_str = f'episode: {episode}, steps: {steps} , score: {score:.2f} ,avg_score:{average_score:.2f}, alpha:{agent.alpha.numpy():.2f}'
        f.write(log_str+'\n')
        t = int(time.time()-start_time)
        print(log_str + f' time: {t//3600:2}:{t%3600//60:2}:{t%60:2}') 
        if average_score > best_score:
            best_score = average_score
            agent.save_models()

except Exception as e: 
    print ('error is ', e)

finally:
    f.close()
    plt.plot(np.array(score_list),label='score')
    plt.plot(np.array(average_score_list),label='average score')
    plt.show()
    env.close()