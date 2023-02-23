import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import nest_utils
import functools
#--------------------------------------------------------------------
class ConvLayer(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch, 64, 64, 3], outputs are of shape [batch, 1, 1, 64]
	def __init__(self, conv1_filter=8, conv1_kernel=5, conv1_stride=2, conv1_padding='SAME', \
			conv2_filter=16,  conv2_kernel=3, conv2_stride=2, conv2_padding='SAME', \
			conv3_filter=32,  conv3_kernel=3, conv3_stride=2, conv3_padding='SAME', \
			conv4_filter=64,  conv4_kernel=3, conv4_stride=2, conv4_padding='SAME', \
			conv5_filter=64,  conv5_kernel=4, conv5_stride=1, conv5_padding='VALID', \
			name='conv_layer'):
		super(ConvLayer, self).__init__()
		self.conv1 = keras.layers.Conv2D(conv1_filter, conv1_kernel, conv1_stride, conv1_padding, activation=tf.nn.leaky_relu)
		self.conv2 = keras.layers.Conv2D(conv2_filter, conv2_kernel, conv2_stride, conv2_padding, activation=tf.nn.leaky_relu)
		self.conv3 = keras.layers.Conv2D(conv3_filter, conv3_kernel, conv3_stride, conv3_padding, activation=tf.nn.leaky_relu)
		self.conv4 = keras.layers.Conv2D(conv4_filter, conv4_kernel, conv4_stride, conv4_padding, activation=tf.nn.leaky_relu)
		self.conv5 = keras.layers.Conv2D(conv5_filter, conv5_kernel, conv5_stride, conv5_padding, activation=tf.nn.leaky_relu)
	#----------------------------------------------------------------
	def call(self, inputs):
		conv1_output = self.conv1(inputs)
		conv2_output = self.conv2(conv1_output)
		conv3_output = self.conv3(conv2_output)
		conv4_output = self.conv4(conv3_output)
		output = self.conv5(conv4_output)
		return output
#--------------------------------------------------------------------
class TransConvLayer(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch, 1, 1, depth], outputs are of shape [batch, 64, 64, 3]
	def __init__(self, transconv1_filter=64, transconv1_kernel=4, transconv1_stride=1, transconv1_padding='VALID', \
			transconv2_filter=32,  transconv2_kernel=3, transconv2_stride=2, transconv2_padding='SAME', \
			transconv3_filter=16,  transconv3_kernel=3, transconv3_stride=2, transconv3_padding='SAME', \
			transconv4_filter=8,  transconv4_kernel=3, transconv4_stride=2, transconv4_padding='SAME', \
			transconv5_filter=3, transconv5_kernel=5, transconv5_stride=2, transconv5_padding='SAME', \
			name='trans_conv_layer'):
		super(TransConvLayer, self).__init__()
		self.transconv1 = keras.layers.Conv2DTranspose(transconv1_filter, transconv1_kernel, transconv1_stride, transconv1_padding, activation=tf.nn.leaky_relu)
		self.transconv2 = keras.layers.Conv2DTranspose(transconv2_filter, transconv2_kernel, transconv2_stride, transconv2_padding, activation=tf.nn.leaky_relu)
		self.transconv3 = keras.layers.Conv2DTranspose(transconv3_filter, transconv3_kernel, transconv3_stride, transconv3_padding, activation=tf.nn.leaky_relu)
		self.transconv4 = keras.layers.Conv2DTranspose(transconv4_filter, transconv4_kernel, transconv4_stride, transconv4_padding, activation=tf.nn.leaky_relu)
		self.transconv5 = keras.layers.Conv2DTranspose(transconv5_filter, transconv5_kernel, transconv5_stride, transconv5_padding, activation=tf.nn.leaky_relu)
	#----------------------------------------------------------------
	def call(self, inputs):
		transconv1_output = self.transconv1(inputs)
		transconv2_output = self.transconv2(transconv1_output)
		transconv3_output = self.transconv3(transconv2_output)
		transconv4_output = self.transconv4(transconv3_output)
		output = self.transconv5(transconv4_output)
		return output
#--------------------------------------------------------------------
class CriticNetwork(keras.Model):
	def __init__(self, fc1_dims=256, fc2_dims=256, \
			name='critic', chkpt_dir='tmp/critic'):
		super(CriticNetwork, self).__init__()
		#------------------------------------------------------------
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
		#------------------------------------------------------------
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu', name='fc1')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu', name='fc2')
		self.critic = keras.layers.Dense(1, activation=None, name='out')
	#----------------------------------------------------------------
	def call(self, latent1, latent2, action):
		fc1_output = self.fc1(tf.concat([latent1,action], axis=1))
		fc2_output = self.fc2(fc1_output)
		#------------------------------------------------------------
		critic = self.critic(fc2_output)
		#------------------------------------------------------------
		return critic
#--------------------------------------------------------------------
class ActorNetwork(keras.Model):
	def __init__(self, action_shape=(3,), \
			fc1_dims=256, fc2_dims=256, \
			name='actor', chkpt_dir='tmp/actor'):
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
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation='relu', name='fc1')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation='relu', name='fc2')
		self.mu = keras.layers.Dense(self.action_shape, activation=None, name='mu')
		self.sigma = keras.layers.Dense(self.action_shape, activation='sigmoid', name='sigma')
	#----------------------------------------------------------------
	def call(self, features, actions):
		inputs = tf.concat([features,actions], axis=-1)
		fc1_output = self.fc1(inputs)
		fc2_output = self.fc2(fc1_output)
		mu = self.mu(fc2_output)
		sigma = self.sigma(fc2_output)
		#------------------------------------------------------------
		return mu, sigma
	#----------------------------------------------------------------
	def sample_normal(self, features, actions, reparameterize=True):
		mu, sigma = self.call(features, actions)
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
#--------------------------------------------------------------------
class MultivariateNormalLayer(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch, base_shape], outputs are of shape [batch, sequence, output_size]
	def __init__(self, output_size=32, sigma=None, fc1_dims=64, fc2_dims=64, name='multi_variate_normal_layer'):
		super(MultivariateNormalLayer, self).__init__()
		self.output_size = output_size
		if sigma == None:
			self.generate_sigma = True
		else:
			self.generate_sigma = False
			self.sigma = sigma
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc1 = keras.layers.Dense(self.fc1_dims, activation=tf.nn.leaky_relu, name='fc1')
		self.fc2 = keras.layers.Dense(self.fc2_dims, activation=tf.nn.leaky_relu, name='fc2')
		self.mu = keras.layers.Dense(self.output_size, activation=None, name='mu')
		if self.generate_sigma:
			self.sigma = keras.layers.Dense(self.output_size, activation='sigmoid', name='sigma')
	#----------------------------------------------------------------
	def call(self, *inputs):
		if len(inputs) > 1:
			inputs = tf.concat(inputs, axis=-1)
		else:
			inputs, = inputs
		fc1_output = self.fc1(inputs)
		fc2_output = self.fc2(fc1_output)
		mu = self.mu(fc2_output)
		if self.generate_sigma:
			sigma = self.sigma(fc2_output)
		else:
			sigma = self.sigma * tf.ones_like(mu)
		return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#--------------------------------------------------------------------
class ConstantMultivariateNormalLayer(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch, base_shape], outputs are of shape [batch, sequence, output_size]
	def __init__(self, output_size=32, sigma=None, ame='const_multi_variate_normal_layer'):
		super(ConstantMultivariateNormalLayer, self).__init__()
		self.output_size = output_size
		self.sigma = sigma
	#----------------------------------------------------------------
	def call(self, inputs):
		batch_shape = tf.shape(inputs)[0]
		sequence = tf.shape(inputs)[1]
		shape = tf.concat([batch_shape, sequence, self.output_size],axis=0)
		mu = tf.zeros(shape)
		if self.sigma is None:
			sigma = tf.ones(shape)
		else:
			sigma = tf.ones(shape) * self.sigma
		return tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#--------------------------------------------------------------------
class Compressor(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch, sequence, 64, 64, 3], outputs are of shape [batch, sequence, 64]
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
		expanded_shape = tf.concat((tf.shape(inputs)[:-3],[self.feature_size]), axis=0)
		return tf.reshape(conv_output, expanded_shape)
#--------------------------------------------------------------------
class Decoder(keras.layers.Layer):
	# in current use case, expected inputs are of shape [batch*sequence, 1, 1, depth], outputs are of shape [batch, sequence, 64, 64, 3]
	def __init__(self,sigma=0.1):
		super(Decoder, self).__init__()
		self.sigma = sigma
		self.transconv = TransConvLayer()
	#----------------------------------------------------------------
	def call(self, *inputs):
		if len(inputs)>1:
			latent = tf.concat(inputs, axis=-1)
		else:
			latent, = inputs
		collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
		reshape_output = tf.reshape(latent, collapsed_shape)
		transconv_output = self.transconv(reshape_output)
		expanded_shape = tf.concat([tf.shape(latent)[:-1], tf.shape(transconv_output)[1:]], axis=0)
		output = tf.reshape(transconv_output, expanded_shape)
		return tfp.distributions.Independent(distribution=tfp.distributions.Normal(loc=output, scale=self.sigma),reinterpreted_batch_ndims=3)
#--------------------------------------------------------------------
class ModelNetwork(keras.Model):
	def __init__(self, observation_shape=(64,64,3), action_shape=(3,), latent1_size=32, latent2_size=256, name='latent_model', chkpt_dir='tmp/latent_model'):
		super(ModelNetwork,self).__init__()
		self.latent1_size = latent1_size
		self.latent2_size = latent2_size
		#------------------------------------------------------------
		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
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

	#----------------------------------------------------------------
	def sample_posterior(self,observations,actions,step_types):
		sequence_length = step_types.shape[1] - 1
		actions = actions[:, :sequence_length]
		features = self.compressor(observations)
		#------------------------------------------------------------
		# swap batch and time axes
		features = tf.transpose(features, [1, 0, 2])
		actions = tf.transpose(actions, [1, 0, 2])
		step_types = tf.transpose(step_types, [1, 0, 2])
		#------------------------------------------------------------
		latent1_dists = []
		latent1_samples = []
		latent2_dists = []
		latent2_samples = []
		for t in range(sequence_length + 1):
			if t == 0:
				latent1_dist = self.latent1_first_posterior(features[t])
				latent1_sample = latent1_dist.sample()
				latent2_dist = self.latent2_first_posterior(latent1_sample)
				latent2_sample = latent2_dist.sample()
			else:
				reset_mask = tf.equal(step_types[t], 0)
				latent1_first_dist = self.latent1_first_posterior(features[t])
				latent1_dist = self.latent1_posterior(features[t],latent2_samples[t-1],actions[t-1])
				latent1_dist = nest_utils.map_distribution_structure(functools.partial(tf.where, reset_mask), latent1_first_dist, latent1_dist)
				latent1_sample = latent1_dist.sample()
				#----------------------------------------------------
				latent2_first_dist = self.latent2_first_posterior(latent1_sample)
				latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
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
	def compute_loss(self, observations, actions, step_types):
		sequence_length = step_types.shape[1] - 1
		#------------------------------------------------------------
		(latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = self.sample_posterior(observations, actions, step_types)
		#------------------------------------------------------------
		def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
			after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
			prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
			return prior_tensors
		#------------------------------------------------------------
		reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1],dtype=tf.bool),tf.equal(step_types[:,1:],0)],axis=1)
		#------------------------------------------------------------
		latent1_reset_masks = tf.tile(reset_masks, [1, 1, self.latent1_size])
		latent1_first_prior_dists = self.latent1_first_prior(step_types)
		# these distributions start at t=1 and the inputs are from t-1
		latent1_after_first_prior_dists = self.latent1_prior(latent2_posterior_samples[:, :sequence_length],actions[:, :sequence_length])
		latent1_prior_dists = nest_utils.map_distribution_structure(functools.partial(where_and_concat, latent1_reset_masks),latent1_first_prior_dists,latent1_after_first_prior_dists)
		#------------------------------------------------------------
		latent2_reset_masks = tf.tile(reset_masks, [1, 1, self.latent2_size])
		latent2_first_prior_dists = self.latent2_first_prior(latent1_posterior_samples)
		# these distributions start at t=1 and the last 2 inputs are from t-1
		latent2_after_first_prior_dists = self.latent2_prior(latent1_posterior_samples[:, 1:sequence_length+1],latent2_posterior_samples[:, :sequence_length],actions[:, :sequence_length])
		latent2_prior_dists = nest_utils.map_distribution_structure(functools.partial(where_and_concat, latent2_reset_masks),latent2_first_prior_dists,latent2_after_first_prior_dists)
		#------------------------------------------------------------
		latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples) - latent1_prior_dists.log_prob(latent1_posterior_samples))
		latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
		if self.latent2_posterior == self.latent2_prior:
			latent2_kl_divergences = 0.0
		else:
			latent2_kl_divergences = (latent2_posterior_dists.log_prob(latent2_posterior_samples)- latent2_prior_dists.log_prob(latent2_posterior_samples))
			latent2_kl_divergences = tf.reduce_sum(latent2_kl_divergences, axis=1)
		likelihood_dists = self.decoder(latent1_posterior_samples, latent2_posterior_samples)
		likelihood_log_probs = likelihood_dists.log_prob(observations)
		likelihood_log_probs = tf.reduce_sum(likelihood_log_probs, axis=1)
		reconstruction_error = tf.reduce_sum(tf.square(observations - likelihood_dists.distribution.loc),axis=list(range(-len(likelihood_dists.event_shape), 0)))
		reconstruction_error = tf.reduce_sum(reconstruction_error, axis=1)
		# summed over the time dimension
		elbo = likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences
		# average over the batch dimension
		loss = -tf.reduce_mean(elbo)
		return loss
