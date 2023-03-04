import gym
from gym.spaces import Box
import numpy as np
import tensorflow as tf

class WrappedGymEnv(gym.Wrapper):
	def __init__(self, env, image_size=[64,64], steps_com=1):
		super(WrappedGymEnv,self).__init__(env)
		self.env = env
		self.image_size = image_size
		self.steps_com = steps_com
		
		self.action_space = Box(-1.0, 1.0, \
        shape=(2,))
		
		self.action_con_low = \
		self.action_space.low.astype("float32")
		self.action_con_high = \
		self.action_space.high.astype("float32")
		
		self.observation_low = 0.0
		self.observation_high = 255.0
		
		self.num_step = 0
	def reset(self):
		self.num_step = 0
		obs_raw = self.env.reset()
		obs = {
        'bev' : tf.image.resize(tf.convert_to_tensor(obs_raw['bev']), \
        self.image_size).numpy().astype(np.float32) / self.observation_high,
        }
		return obs
	def step(self, action):
		act = self.action_con_low + (action + 1.0) * 0.5 * \
		(self.action_con_high - self.action_con_low)
		
		if action[0] > 0:
			throttle = np.clip(action[0],0.0,1.0)
			brake = 0
		else:
			throttle = 0
			brake = np.clip(-action[0],0.0,1.0)
		
		act_tuple = ([throttle, brake, action[1]], [False])
		
		total_reward = 0
		for _ in range(self.steps_com):
			obs_raw, reward, done, info = self.env.step(act_tuple)
			total_reward += reward
			if done:
				break
		# 2 means last step, 1 means middle step, 0 means first step
		if done:
			step_type = 2
		elif self.num_step > 1:
			step_type = 1
		else:
			step_type = 0
		self.num_step += 1
		obs = {
        'bev' : tf.image.resize(tf.convert_to_tensor(obs_raw['bev']), \
        self.image_size).numpy().astype(np.float32) / self.observation_high,
        }
		return obs, np.array([total_reward]), np.array([step_type]), \
		np.array([done]), info
