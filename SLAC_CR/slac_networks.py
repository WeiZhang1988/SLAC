import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import nest_utils
import functools
#--------------------------------------------------------------------
class ConvLayer(keras.layers.Layer):
	def __init__(self, conv1_filter=8, conv1_kernel=5, conv1_stride=2, conv1_padding='SAME', \
			conv2_filter=16,  conv2_kernel=3, conv2_stride=2, conv2_padding='SAME', \
			conv3_filter=32,  conv3_kernel=3, conv3_stride=2, conv3_padding='SAME', \
			conv4_filter=64,  conv4_kernel=3, conv4_stride=2, conv4_padding='SAME', \
			conv5_filter=64,  conv5_kernel=4, conv5_stride=1, conv5_padding='VALID'):
		super(ConvLayer, self).__init__()
		self.conv1 = keras.layers.Conv2D(conv1_filter, conv1_kernel, conv1_stride, conv1_padding, activation=tf.nn.leaky_relu)
		self.conv2 = keras.layers.Conv2D(conv2_filter, conv2_kernel, conv2_stride, conv2_padding, activation=tf.nn.leaky_relu)
		self.conv3 = keras.layers.Conv2D(conv3_filter, conv3_kernel, conv3_stride, conv3_padding, activation=tf.nn.leaky_relu)
		self.conv4 = keras.layers.Conv2D(conv4_filter, conv4_kernel, conv4_stride, conv4_padding, activation=tf.nn.leaky_relu)
		self.conv5 = keras.layers.Conv2D(conv5_filter, conv5_kernel, conv5_stride, conv5_padding, activation=tf.nn.leaky_relu)
	#----------------------------------------------------------------
	def call(self, observation):
		conv1_output = self.conv1(observation)
		conv2_output = self.conv2(conv1_output)
		conv3_output = self.conv3(conv2_output)
		conv4_output = self.conv4(conv3_output)
		output = self.conv5(conv4_output)
		return output
#--------------------------------------------------------------------
class TransConvLayer(keras.layers.Layer):
	def __init__(self, transconv1_filter=64, transconv1_kernel=4, transconv1_stride=1, transconv1_padding='VALID', \
			transconv2_filter=32,  transconv2_kernel=3, transconv2_stride=2, transconv2_padding='SAME', \
			transconv3_filter=16,  transconv3_kernel=3, transconv3_stride=2, transconv3_padding='SAME', \
			transconv4_filter=8,  transconv4_kernel=3, transconv4_stride=2, transconv4_padding='SAME', \
			transconv5_filter=3, transconv5_kernel=5, transconv5_stride=2, transconv5_padding='SAME'):
#--------------------------------------------------------------------
class CriticNetwork(keras.Model):
	def __init__(self, fc1_dims=256, fc2_dims=256, \
			name='critic', chkpt_dir='tmp/slac'):
		super(CriticNetwork, self).__init__()
		#------------------------------------------------------------
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.critic = keras.layers.Dense(1, activation=None)
	#----------------------------------------------------------------
	def call(self, state, action):
		fc1_output = self.fc1(tf.concat([state,action], axis=1))
		fc2_output = self.fc2(fc1_output)
		#------------------------------------------------------------
		critic = self.critic(fc2_output)
		#------------------------------------------------------------
		return critic
#--------------------------------------------------------------------
class ActorNetwork(keras.Model):
	def __init__(self, action_shape=(3,), \
			fc1_dims=256, fc2_dims=256, \
			name='actor', chkpt_dir='tmp/slac'):
		super(ActorNetwork, self).__init__()
		#------------------------------------------------------------
		self.action_shape = action_shape[0]
		#------------------------------------------------------------
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.noise = 1e-6
		#------------------------------------------------------------
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu')
		self.mu = keras.layers.Dense(self.action_shape, activation=None)
		self.sigma = keras.layers.Dense(self.action_shape, activation='sigmoid')
	#----------------------------------------------------------------
	def call(self, feature):
		fc1_output = self.fc1(feature)
		fc2_output = self.fc2(fc1_output)
		mu = self.mu(fc2_output)
		sigma = self.sigma(fc2_output)
		#------------------------------------------------------------
		return mu, sigma
	#----------------------------------------------------------------
	def sample_normal(self, feature, reparameterize=True):
		mu, sigma = self.call(feature)
		probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
		#------------------------------------------------------------
		if reparameterize:
			probabilities_std = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(sigma))
			epsilon = probabilities_std.sample()
			action = epsilon*sigma + mu
		else:
			action = probabilities.sample()
		log_prob = probabilities.log_prob(action)
		#------------------------------------------------------------
		action = tf.math.tanh(action)
		log_probs = tf.math.log(1-tf.math.pow(action,2)+self.noise)
		log_prob -= tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
		#------------------------------------------------------------
		return action, log_prob
#--------------------------------------------------------------------
class MultivariateNormalLayer(keras.layers.Layer):
	def __init__(self, output_size=32, sigma=None, fc1_dims=64, fc2_dims=64):
		super(MultivariateNormalLayer, self).__init__()
		self.output_size = output_size
		if sigma = None:
			self.generate_sigma = True
		else:
			self.generate_sigma = False
			self.sigma = sigma
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation=tf.nn.leaky_relu)
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation=tf.nn.leaky_relu)
		self.mu = keras.layers.Dense(self.output_size, activation=None)
		if self.generate_sigma:
			self.sigma = keras.layers.Dense(self.output_size, activation='sigmoid')
	#----------------------------------------------------------------
	def call(self, inputs):
		fc1_output = self.fc1(inputs)
		fc2_output = self.fc2(fc1_output)
		mu = self.mu(fc2_output)
		if self.generate_sigma
			sigma = self.sigma(fc2_output)
		else:
			sigma = self.sigma * tf.ones_like(mu)
		return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#--------------------------------------------------------------------
class ConstantMultivariateNormalLayer(keras.layers.Layer):
	def __init__(self, output_size=32, sigma=None):
		super(ConstantMultivariateNormalLayer, self).__init__()
		self.output_size = output_size
		self.sigma = sigma
	#----------------------------------------------------------------
	def call(self, inputs):
		batch_shape = tf.shape(inputs[0])
		shape = tf.concat([batch_shape, output_size],axis=0)
		mu = tf.zeros(shape)
		if self.sigma is None:
			sigma = tf.ones(shape)
		else:
			sigma = tf.ones(shape) * self.sigma
		return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#--------------------------------------------------------------------
class Compressor(keras.layers.Layer):
	def __init__(self, feature_size=64):
		super(Compressor, self).__init__()
		self.feature_size = feature_size
		self.conv = ConvLayer()
	#----------------------------------------------------------------
	def call(self, inputs):
		input_shape = tf.shape(inputs)[-3:]
		collapsed_shape = tf.concat(([-1], input_shape), axis=0)
		reshape_output = tf.reshape(inputs, collapsed_shape)
		conv_output = self.conv(reshape_output)
		expanded_shape = tf.concat((tf.shape(inputs)[:-3][self.feature_size]), axis=0)
		return tf.reshape(conv_output, expanded_shape)
#--------------------------------------------------------------------
class Decoder(keras.layers.Layer):
	def __init__(self,sigma=1.0):
		super(Decoder, self).__init__()
		self.sigma = sigma
		self.transconv = TransConvLayer
	#----------------------------------------------------------------
	def call(self, inputs):
		collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
		reshape_output = tf.reshape(inputs, collapsed_shape)
		transconv_output = self.transconv(reshape_output)
		expanded_shape = tf.concat([tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
		output = tf.reshape(transconv_output, expanded_shape)
		return tfp.distributions.Independent(distribution=tfp.distributions.Normal(loc=output, scale=self.sigma),reinterpreted_batch_ndims=3)
#--------------------------------------------------------------------
class ModelNetwork(keras.Model):
	def __init__(self, observation_shape=(96,96,3), action_shape=(3,), latent1_size=32, latent2_size=256):
		super(ModelNetwork,self).__init__()
		self.latent1_size = latent1_size
		self.latent2_size = latent2_size
		#------------------------------------------------------------
		self.compressor = Compressor()
		self.decoder = Decoder()
		#------------------------------------------------------------
		# p(z_1^1)
		self.latent1_first_prior = ConstantMultivariateNormalLayer(self.latent1_size)
		# p(z_1^2 | z_1^1)
		self.latent2_first_prior = MultivariateNormalLayer(self.latent2_size)
		# p(z_{t+1}^1 | z_t^2, a_t)
		self.latent1_prior = MultivariateNormalLayer(self.latent1_size)
		# p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
		self.latent2_prior = MultivariateNormalLayer(self.latent2_size)
		#------------------------------------------------------------
		# q(z_1^1 | x_1)
		self.latent1_first_posterior = MultivariateNormalLayer(self.latent1_size)
		# q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
		self.latent2_first_posterior = self.latent2_first_prior
		# q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
		self.latent1_posterior = MultivariateNormalLayer(self.latent1_size)
		# q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
		self.latent2_posterior = self.latent2_prior
	#----------------------------------------------------------------
	def sample_posterior(self,observation,action,step_type):
		sequence_length = step_types.shape[1].value - 1
		action = action[:, :sequence_length]
		feature = self.compressor(observation)
		#------------------------------------------------------------
		# swap batch and time axes
		feature = tf.transpose(feature, [1, 0, 2])
		action = tf.transpose(action, [1, 0, 2])
		step_type = tf.transpose(step_type, [1, 0])
		#------------------------------------------------------------
		latent1_dists = []
		latent1_samples = []
		latent2_dists = []
		latent2_samples = []
		for t in range(sequence_length + 1):
			if t == 0:
				latent1_dist = self.latent1_first_posterior(feature[t])
				latent1_sample = latent1_dist.sample()
				latent2_dist = self.latent2_first_posterior(latent1_sample)
				latent2_sample = latent2_dist.sample()
			else:
				reset_mask = tf.equal(step_type[t], 0)
				latent1_first_dist = self.latent1_first_posterior(feature[t])
				latent1_dist = self.latent1_posterior(feature[t],latent2_samples[t-1],action[t-1])
				latent1_dist = nest_utils.map_distribution_structure(functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
				latent1_sample = latent1_dist.sample()
				#----------------------------------------------------
				latent2_first_dist = self.latent2_first_posterior(latent1_sample)
				latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], action[t-1])
				latent2_dist = nest_utils.map_distribution_structure(functools.partial(tf.where, reset_mask), latent2_first_dist, latent2_dist)
				latent2_sample = latent2_dist.sample()
			#--------------------------------------------------------
			latent1_dists.append(latent1_dist)
			latent1_samples.append(latent1_sample)
			latent2_dists.append(latent2_dist)
			latent2_samples.append(latent2_sample)
		#------------------------------------------------------------	
		latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
		latent1_samples = tf.stack(latent1_samples, axis=1)
		latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
		latent2_samples = tf.stack(latent2_samples, axis=1)
		#------------------------------------------------------------
		return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
	#----------------------------------------------------------------
	def compute_loss(self,):
