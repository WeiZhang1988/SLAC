import gym
from gym.spaces import Box
import numpy as np
import tensorflow as tf

class WrappedGymEnv(gym.Wrapper):
	def __init__(self, env, steps_com=1):
		super(WrappedGymEnv,self).__init__(env)
		self.env = env
		self.steps_com = steps_com
		
		self.action_space = Box(np.array([0.0, 0.0, -1.0]), 1.0, \
        shape=(3,))
		
		self.action_con_low = \
		self.action_space.low.astype("float32")
		self.action_con_high = \
		self.action_space.high.astype("float32")
		
		self.observation_lc_low = \
		self.env.observation_space['left_camera'].low.astype("float32")
		self.observation_lc_high = \
		self.env.observation_space['left_camera'].high.astype("float32")
		
		self.observation_fc_low = \
		self.env.observation_space['front_camera'].low.astype("float32")
		self.observation_fc_high = \
		self.env.observation_space['front_camera'].high.astype("float32")
		
		self.observation_rtc_low = \
		self.env.observation_space['right_camera'].low.astype("float32")
		self.observation_rtc_high = \
		self.env.observation_space['right_camera'].high.astype("float32")
		
		self.observation_rrc_low = \
		self.env.observation_space['rear_camera'].low.astype("float32")
		self.observation_rrc_high = \
		self.env.observation_space['rear_camera'].high.astype("float32")
		
		self.observation_li_low = \
		self.env.observation_space['lidar_image'].low.astype("float32")
		self.observation_li_high = \
		self.env.observation_space['lidar_image'].high.astype("float32")
		
		self.observation_ri_low = \
		self.env.observation_space['radar_image'].low.astype("float32")
		self.observation_ri_high = \
		self.env.observation_space['radar_image'].high.astype("float32")
		
		
		
		self.num_step = 0
	def reset(self):
		self.num_step = 0
		obs_raw = self.env.reset()
		obs = {
        'left_camera' : tf.image.resize(obs_raw['left_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'front_camera': tf.image.resize(obs_raw['front_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'right_camera': tf.image.resize(obs_raw['right_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'rear_camera' : tf.image.resize(obs_raw['rear_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'lidar_image' : tf.image.resize(obs_raw['lidar_image'], [96,96]).numpy().astype(np.float32) / 255.0,
        'radar_image' : tf.image.resize(obs_raw['radar_image'], [96,96]).numpy().astype(np.float32) / 255.0,
        'gnss': obs_raw['gnss'],
        'imu' : obs_raw['imu'],
        'trgt_pos' : obs_raw['trgt_pos'],
        }
		return obs
	def step(self, action):
		act = self.action_con_low + (action + 1.0) * 0.5 * \
		(self.action_con_high - self.action_con_low)
		act_tuple = (np.clip(act, self.action_con_low, \
		self.action_con_high), [False])
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
        'left_camera' : tf.image.resize(obs_raw['left_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'front_camera': tf.image.resize(obs_raw['front_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'right_camera': tf.image.resize(obs_raw['right_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'rear_camera' : tf.image.resize(obs_raw['rear_camera'], [96,96]).numpy().astype(np.float32) / 255.0,
        'lidar_image' : tf.image.resize(obs_raw['lidar_image'], [96,96]).numpy().astype(np.float32) / 255.0,
        'radar_image' : tf.image.resize(obs_raw['radar_image'], [96,96]).numpy().astype(np.float32) / 255.0,
        'gnss': obs_raw['gnss'],
        'imu' : obs_raw['imu'],
        'trgt_pos' : obs_raw['trgt_pos'],
        }
		return obs, np.array([total_reward]), np.array([step_type]), \
		np.array([done]), info
