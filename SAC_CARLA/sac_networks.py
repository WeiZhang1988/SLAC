import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
#--------------------------------------------------------------------
class ConvLayer(keras.layers.Layer):
	def __init__(self, conv1_filter=8, conv1_kernel=4, conv1_stride=2, conv1_padding='SAME', \
			conv2_filter=16,  conv2_kernel=3, conv2_stride=2, conv2_padding='SAME', \
			conv3_filter=32,  conv3_kernel=3, conv3_stride=2, conv3_padding='SAME', \
			conv4_filter=64,  conv4_kernel=3, conv4_stride=2, conv4_padding='SAME', \
			conv5_filter=128, conv5_kernel=3, conv5_stride=1, conv5_padding='SAME', \
			conv6_filter=256, conv6_kernel=3, conv6_stride=1, conv6_padding='SAME'):
		super(ConvLayer, self).__init__()
		self.conv1 = keras.layers.Conv2D(conv1_filter, conv1_kernel, conv1_stride, conv1_padding, activation=tf.nn.leaky_relu)
		self.conv2 = keras.layers.Conv2D(conv2_filter, conv2_kernel, conv2_stride, conv2_padding, activation=tf.nn.leaky_relu)
		self.conv3 = keras.layers.Conv2D(conv3_filter, conv3_kernel, conv3_stride, conv3_padding, activation=tf.nn.leaky_relu)
		self.conv4 = keras.layers.Conv2D(conv4_filter, conv4_kernel, conv4_stride, conv4_padding, activation=tf.nn.leaky_relu)
		self.conv5 = keras.layers.Conv2D(conv5_filter, conv5_kernel, conv5_stride, conv5_padding, activation=tf.nn.leaky_relu)
		self.conv6 = keras.layers.Conv2D(conv6_filter, conv6_kernel, conv6_stride, conv6_padding, activation=tf.nn.leaky_relu)
		self.flat = keras.layers.Flatten()
	def call(self, observation):
		conv1_output = self.conv1(observation)
		conv2_output = self.conv2(conv1_output)
		conv3_output = self.conv3(conv2_output)
		conv4_output = self.conv4(conv3_output)
		conv5_output = self.conv5(conv4_output)
		conv6_output = self.conv6(conv5_output)
		output = self.flat(conv6_output)
		return output
#--------------------------------------------------------------------
class CriticNetwork(keras.Model):
	def __init__(self, fc1_dims=512, fc2_dims=256, fc3_dims=128, \
			name='critic', chkpt_dir='tmp/critic'):
		super(CriticNetwork, self).__init__()
		#------------------------------------------------------------
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.image_input = keras.layers.InputLayer(input_shape=(180,180,3))
		self.action_input = keras.layers.InputLayer(input_shape=(3,))
		#------------------------------------------------------------
		self.conv = ConvLayer()
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.fc3 = keras.layers.Dense(self.fc3_dims, activation='relu')
		self.critic = keras.layers.Dense(1, activation=None)
	#----------------------------------------------------------------
	def call(self, image, action):
		image_output = self.image_input(image)
		action_output = self.action_input(action)
		conv_output = self.conv(image_output)
		fc1_output = self.fc1(tf.concat([conv_output, action], axis=1))
		fc2_output = self.fc2(fc1_output)
		fc3_output = self.fc3(fc2_output)
		#------------------------------------------------------------
		critic = self.critic(fc3_output)
		#------------------------------------------------------------
		return critic
#--------------------------------------------------------------------
class ActorNetwork(keras.Model):
	def __init__(self, action_shape=(2,), \
			fc1_dims=512, fc2_dims=256, fc3_dims=128, \
			name='actor', chkpt_dir='tmp/actor'):
		super(ActorNetwork, self).__init__()
		#------------------------------------------------------------
		self.action_shape = action_shape[0]
		#------------------------------------------------------------
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.noise = 1e-6
		#------------------------------------------------------------
		self.image_input = keras.layers.InputLayer(input_shape=(180,180,3))
		#------------------------------------------------------------
		self.conv = ConvLayer()
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.fc3 = keras.layers.Dense(self.fc3_dims, activation='relu')
		self.mu = keras.layers.Dense(self.action_shape, activation=None)
		self.sigma = keras.layers.Dense(self.action_shape, activation='sigmoid')
	#----------------------------------------------------------------
	def call(self, image):
		image_output = self.image_input(image)
		conv_output = self.conv(image_output)
		fc1_output = self.fc1(tf.concat([conv_output], axis=1))
		fc2_output = self.fc2(fc1_output)
		fc3_output = self.fc3(fc2_output)
		mu = self.mu(fc3_output)
		sigma = self.sigma(fc3_output)
		#------------------------------------------------------------
		return mu, sigma
	#----------------------------------------------------------------
	def sample_normal(self, image, reparameterize=True):
		mu, sigma = self.call(image)
		probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
		#------------------------------------------------------------
		if reparameterize:
			probabilities_std = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(sigma))
			epsilon = probabilities_std.sample()
			action = epsilon*sigma + mu
		else:
			action = probabilities.sample()
		log_prob = probabilities.log_prob(action)
		log_prob = tf.expand_dims(log_prob,axis=-1)
		#------------------------------------------------------------
		action = tf.math.tanh(action)
		log_probs = tf.math.log(1-tf.math.pow(action,2)+self.noise)
		log_prob -= tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
		#------------------------------------------------------------
		return action, log_prob
