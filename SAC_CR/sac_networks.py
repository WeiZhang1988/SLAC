import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
#--------------------------------------------------------------------
class CriticNetwork(keras.Model):
	def __init__(self, conv1_filter=64, conv1_kernel=5, conv1_stride=2, conv1_padding="SAME", \
			conv2_filter=32, conv2_kernel=3, conv2_stride=1, conv2_padding="VALID", \
			state_fc_dims=256, action_fc_dims=256, joint_fc_dims=256, \
			name='critic', chkpt_dir='tmp/sac'):
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
		self.state_fc_dims = state_fc_dims
		self.action_fc_dims = action_fc_dims
		self.joint_fc_dims = joint_fc_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.conv1 = keras.layers.Conv2D(self.conv1_filter, self.conv1_kernel, self.conv1_stride, self.conv1_padding, activation=tf.nn.leaky_relu)
		self.conv2 = keras.layers.Conv2D(self.conv2_filter, self.conv2_kernel, self.conv2_stride, self.conv2_padding, activation=tf.nn.leaky_relu)
		self.flat = keras.layers.Flatten()
		self.state_fc = keras.layers.Dense(self.state_fc_dims, activation='relu')
		self.action_fc = keras.layers.Dense(self.action_fc_dims, activation='relu')
		self.joint_fc = keras.layers.Dense(self.joint_fc_dims, activation='relu')
		self.critic = keras.layers.Dense(1, activation=None)
	#----------------------------------------------------------------
	def call(self, state, action):
		conv1_output = self.conv1(state)
		conv2_output = self.conv2(conv1_output)
		flat_output = self.flat(conv2_output)
		state_fc_output = self.state_fc(flat_output)
		action_fc_output = self.action_fc(action)
		# concatenate in axis=1 because axis=0 is batch dimension
		joint_fc_output = self.joint_fc(tf.concat([state_fc_output, action_fc_output], axis=1))
		#------------------------------------------------------------
		critic = self.critic(joint_fc_output)
		#------------------------------------------------------------
		return critic
#--------------------------------------------------------------------
class ActorNetwork(keras.Model):
	def __init__(self, action_shape=(3,), conv1_filter=64, conv1_kernel=5, conv1_stride=2, conv1_padding="SAME", \
			conv2_filter=32, conv2_kernel=3, conv2_stride=1, conv2_padding="VALID", \
			fc1_dims=256, fc2_dims=256, \
			name='actor', chkpt_dir='tmp/sac'):
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
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.noise = 1e-6
		#------------------------------------------------------------
		self.conv1 = keras.layers.Conv2D(self.conv1_filter, self.conv1_kernel, self.conv1_stride, self.conv1_padding, activation=tf.nn.leaky_relu)
		self.conv2 = keras.layers.Conv2D(self.conv2_filter, self.conv2_kernel, self.conv2_stride, self.conv2_padding, activation=tf.nn.leaky_relu)
		self.flat = keras.layers.Flatten()
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.mu = keras.layers.Dense(self.action_shape, activation=None)
		self.sigma = keras.layers.Dense(self.action_shape, activation='sigmoid')
	#----------------------------------------------------------------
	def call(self, observation):
		conv1_output = self.conv1(observation)
		conv2_output = self.conv2(conv1_output)
		flat_output = self.flat(conv2_output)
		fc1_output = self.fc1(flat_output)
		fc2_output = self.fc2(fc1_output)
		mu = self.mu(fc2_output)
		sigma = self.sigma(fc2_output)
		#------------------------------------------------------------
		return mu, sigma
	#----------------------------------------------------------------
	def sample_normal(self, observation, reparameterize=True):
		mu, sigma = self.call(observation)
		#------------------------------------------------------------
		if reparameterize:
			probabilities = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(sigma))
			epsilon = probabilities.sample()
			log_prob = probabilities.log_prob(epsilon)
			action = epsilon*sigma + mu
		else:
			probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
			action = probabilities.sample()
			log_prob = probabilities.log_prob(action)
		#------------------------------------------------------------
		action = tf.math.tanh(action)
		log_probs = tf.math.log(1-tf.math.pow(action,2)+self.noise) # notice: might incorrect here
		log_prob -= tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
		#------------------------------------------------------------
		return action, log_prob
