import gym
import numpy as np

def normalizeAction(action, action_space):
	action = action_space.low + (action + 1.0) * 0.5 * (action_space.high - action_space.low)
	return np.clip(action, action_space.low, action_space.high)
#--------------------------------------------------------------------
def normalizeObservation(observation, observation_space):
	return np.array(observation/(observation_space.high - observation_space.low)).astype("float32")
#--------------------------------------------------------------------
class ActionObservationNormalizer(object):
	def __init__(self, action_space, observation_space):
		self.action_low = action_space.low
		self.action_high = action_space.high
		self.observation_low = observation_space.low
		self.observation_high = observation_space.high
	def normalizeAction(self,action):
		act = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
		act = np.clip(act, self.action_low, self.action_high)
		return act
	def normalizeObservation(self,observation):
		obs = np.array(observation/(self.observation_high)).astype("float32")
		return obs
#--------------------------------------------------------------------
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
