import gym
import numpy as np

class WrappedGymEnv(gym.Wrapper):
	def __init__(self, env, steps_com):
		super(WrappedGymEnv,self).__init__(env)
		self.env = env
		self.steps_com = steps_com
		self.action_low = self.env.action_space.low.astype("float32")
		self.action_high = self.env.action_space.high.astype("float32")
		self.observation_low = self.env.observation_space.low.astype("float32")
		self.observation_high = self.env.observation_space.high.astype("float32")
		self.num_step = 0
	def reset(self):
		self.num_step = 0
		obs = self.env.reset().astype("float32")/self.observation_high
		obs = obs[16:80,16:80,:]
		return obs
	def step(self, action):
		act = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
		act = np.clip(act, self.action_low, self.action_high)
		total_reward = 0
		for _ in range(self.steps_com):
			observation, reward, done, info = self.env.step(act)
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
		obs = observation.astype("float32")/self.observation_high
		obs = obs[16:80,16:80,:]
		return obs, [total_reward], [step_type], [done], info
