import gym
import numpy as np

class WrappedGymEnv(gym.Wrapper):
	def __init__(self, env):
		super(WrappedGymEnv,self).__init__(env)
		self.env = env
		self.action_low = self.env.action_space.low
		self.action_high = self.env.action_space.high
		self.observation_low = self.env.observation_space.low
		self.observation_high = self.env.observation_space.high
		self.num_step = 0
	def reset(self):
		self.num_step = 0
		obs = self.env.reset()/self.observation_high
		return np.array(obs).astype("float32")
	def step(self, action):
		act = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
		act = np.clip(act, self.action_low, self.action_high)
		observation, reward, done, info = self.env.step(act)
		# 2 means last step, 1 means middle step, 0 means first step
		if done:
			step_type = 2
		elif self.num_step>0:
			step_type = 1
		else:
			step_type = 0
		self.num_step += 1
		obs = observation/self.observation_high
		return np.array(obs).astype("float32"), [reward], [step_type], [done], info
