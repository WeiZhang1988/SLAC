import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
#--------------------------------------------------------------------
class CriticNetwork(keras.Model):
	def __init__(self, conv1_filter=32, conv1_kernel=5, conv1_stride=2, conv1_padding="SAME", \
			conv2_filter=64, conv2_kernel=3, conv2_stride=1, conv2_padding="VALID", \
			fc1_dims=512, fc2_dims=256, fc3_dims=128, \
			name='critic', chkpt_dir='tmp/sac', use_conv=False):
		super(CriticNetwork, self).__init__()
		#------------------------------------------------------------
		self.conv1_filter = conv1_filter
		self.conv1_kernel = conv1_kernel
		self.conv1_stride = conv1_stride
		self.conv1_padding = conv1_padding
		self.conv2_filter = conv2_filter
		self.conv2_kernel = conv2_kernel
		self.conv2_stride = conv2_stride
		self.conv2_padding = conv2_padding
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		self.use_conv = use_conv
		#------------------------------------------------------------
		if self.use_conv:
			self.conv1 = keras.layers.Conv2D(self.conv1_filter, self.conv1_kernel, self.conv1_stride, self.conv1_padding, activation=tf.nn.leaky_relu)
			self.conv2 = keras.layers.Conv2D(self.conv2_filter, self.conv2_kernel, self.conv2_stride, self.conv2_padding, activation=tf.nn.leaky_relu)
			self.flat = keras.layers.Flatten()
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.fc3 = keras.layers.Dense(self.fc3_dims, activation='relu')
		self.critic = keras.layers.Dense(1, activation=None)
	#----------------------------------------------------------------
	def call(self, state, action):
		if self.use_conv:
			conv1_output = self.conv1(state)
			conv2_output = self.conv2(conv1_output)
			flat_output = self.flat(conv2_output)
			fc1_output = self.fc1(tf.concat([flat_output,action], axis=1))
		else:
			fc1_output = self.fc1(tf.concat([state,action], axis=1))
		fc2_output = self.fc2(fc1_output)
		fc3_output = self.fc3(fc2_output)
		#------------------------------------------------------------
		critic = self.critic(fc3_output)
		#------------------------------------------------------------
		return critic
#--------------------------------------------------------------------
class ActorNetwork(keras.Model):
	def __init__(self, action_shape=(1,), conv1_filter=32, conv1_kernel=5, conv1_stride=2, conv1_padding="SAME", \
			conv2_filter=64, conv2_kernel=3, conv2_stride=1, conv2_padding="VALID", \
			fc1_dims=512, fc2_dims=256, fc3_dims=128, \
			name='actor', chkpt_dir='tmp/sac', use_conv = False):
		super(ActorNetwork, self).__init__()
		#------------------------------------------------------------
		self.action_shape = action_shape[0]
		#------------------------------------------------------------
		self.conv1_filter = conv1_filter
		self.conv1_kernel = conv1_kernel
		self.conv1_stride = conv1_stride
		self.conv1_padding = conv1_padding
		self.conv2_filter = conv2_filter
		self.conv2_kernel = conv2_kernel
		self.conv2_stride = conv2_stride
		self.conv2_padding = conv2_padding
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		self.use_conv = use_conv
		#------------------------------------------------------------
		self.noise = 1e-6
		#------------------------------------------------------------
		if self.use_conv:
			self.conv1 = keras.layers.Conv2D(self.conv1_filter, self.conv1_kernel, self.conv1_stride, self.conv1_padding, activation=tf.nn.leaky_relu)
			self.conv2 = keras.layers.Conv2D(self.conv2_filter, self.conv2_kernel, self.conv2_stride, self.conv2_padding, activation=tf.nn.leaky_relu)
			self.flat = keras.layers.Flatten()
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.fc3 = keras.layers.Dense(self.fc3_dims, activation='relu')
		self.mu = keras.layers.Dense(self.action_shape, activation=None)
		self.sigma = keras.layers.Dense(self.action_shape, activation='sigmoid')
	#----------------------------------------------------------------
	def call(self, observation):
		if self.use_conv:
			conv1_output = self.conv1(observation)
			conv2_output = self.conv2(conv1_output)
			flat_output = self.flat(conv2_output)
			fc1_output = self.fc1(flat_output)
		else:
			fc1_output = self.fc1(observation)
		fc2_output = self.fc2(fc1_output)
		fc3_output = self.fc3(fc2_output)
		mu = self.mu(fc3_output)
		sigma = self.sigma(fc3_output)
		#------------------------------------------------------------
		return mu, sigma
	#----------------------------------------------------------------
	def sample_normal(self, observation, reparameterize=True):
		mu, sigma = self.call(observation)
		probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
		#------------------------------------------------------------
		if reparameterize:
			probabilities_std = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(sigma))
			epsilon = probabilities_std.sample()
			action = epsilon*sigma + mu
		else:
			action = probabilities.sample()
		log_prob = probabilities.log_prob(action)
		log_prob = tf.expand_dims(log_prob, axis=-1)
		#------------------------------------------------------------
		action = tf.math.tanh(action)
		log_probs = tf.math.log(1-tf.math.pow(action,2)+self.noise)
		log_prob -= tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
		#------------------------------------------------------------
		return action, log_prob
