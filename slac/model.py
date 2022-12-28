from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import nest_utils
from tensorflow.python.keras import Model

tfd = tfp.distributions

class Normal(tf.Module):
  def __init__(self, base_depth, scale=None, name=None):
    super(Normal, self).__init__(name=name)
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(2 if self.scale is None else 1)

  def __call__(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., 0]
    if self.scale is None:
      assert out.shape[-1].value == 2
      scale = tf.nn.softplus(out[..., 1]) + 1e-5
    else:
      assert out.shape[-1].value == 1
      scale = self.scale
    return tfd.Normal(loc=loc, scale=scale)

class ConstantMultivariateNormalDiag(tf.Module):
  def __init__(self, latent_size, scale=None, name=None):
    super(ConstantMultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.scale = scale

  def __call__(self, inputs):
    # first input should not have any dimensions after the batch_shape, step_type
    batch_shape = tf.shape(inputs)[0] # input is only used to infer batch_shape
    sequence_shape = tf.shape(inputs)[1]
    shape = tf.concat([[batch_shape],[sequence_shape],[self.latent_size]], axis=0)
    loc = tf.zeros(shape)
    if self.scale is None:
      scale_diag = tf.ones(shape)
    else:
      scale_diag = tf.ones(shape) * self.scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

class MultivariateNormalDiag(tf.keras.Model):
  def __init__(self, base_depth, latent_size, scale=None, name=None):
    super(MultivariateNormalDiag, self).__init__(name=name)
    self.latent_size = latent_size
    self.scale = scale
    self.dense1 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.dense2 = tf.keras.layers.Dense(base_depth, activation=tf.nn.leaky_relu)
    self.output_layer = tf.keras.layers.Dense(
        2 * latent_size if self.scale is None else latent_size)

  def call(self, *inputs):
    if len(inputs) > 1:
      inputs = tf.concat(inputs, axis=-1)
    else:
      inputs, = inputs
    out = self.dense1(inputs)
    out = self.dense2(out)
    out = self.output_layer(out)
    loc = out[..., :self.latent_size]
    if self.scale is None:
      assert out.shape[-1] == 2 * self.latent_size
      scale_diag = tf.nn.softplus(out[..., self.latent_size:]) + 1e-5
    else:
      assert out.shape[-1] == self.latent_size
      scale_diag = tf.ones_like(loc) * self.scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

class Decoder(tf.keras.Model):
  """Probabilistic decoder for `p(x_t | z_t)`."""

  def __init__(self, base_depth, channels=3, scale=1.0, name=None):
    super(Decoder, self).__init__(name=name)
    self.scale = scale
    conv_transpose = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation=tf.nn.leaky_relu)
    self.conv_transpose1 = conv_transpose(8 * base_depth, 4, padding="VALID")
    self.conv_transpose2 = conv_transpose(4 * base_depth, 3, 2)
    self.conv_transpose3 = conv_transpose(2 * base_depth, 3, 2)
    self.conv_transpose4 = conv_transpose(base_depth, 3, 2)
    self.conv_transpose5 = conv_transpose(channels, 5, 2)

  def call(self, *inputs):
    if len(inputs) > 1:
      latent = tf.concat(inputs, axis=-1)
    else:
      latent, = inputs
    # (sample, N, T, latent)
    collapsed_shape = tf.stack([-1, 1, 1, tf.shape(latent)[-1]], axis=0)
    out = tf.reshape(latent, collapsed_shape)
    out = self.conv_transpose1(out)
    out = self.conv_transpose2(out)
    out = self.conv_transpose3(out)
    out = self.conv_transpose4(out)
    out = self.conv_transpose5(out)  # (sample*N*T, h, w, c)
    
    expanded_shape = tf.concat(
        [tf.shape(latent)[:-1], tf.shape(out)[1:]], axis=0)
    out = tf.reshape(out, expanded_shape)  # (sample, N, T, h, w, c)
    return tfd.Independent(
        distribution=tfd.Normal(loc=out, scale=self.scale),
        reinterpreted_batch_ndims=3)  # wrap (h, w, c)

class Compressor(tf.keras.Model):
  """Feature extractor."""

  def __init__(self, base_depth, feature_size, name=None):
    super(Compressor, self).__init__(name=name)
    self.feature_size = feature_size
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation='leaky_relu')
    self.conv1 = conv(base_depth, 5, 2)
    self.conv2 = conv(2 * base_depth, 3, 2)
    self.conv3 = conv(4 * base_depth, 3, 2)
    self.conv4 = conv(8 * base_depth, 3, 2)
    self.conv5 = conv(8 * base_depth, 4, padding="VALID")

  def call(self, image):
    image_shape = tf.shape(image)[-3:]
    collapsed_shape = tf.concat(([-1], image_shape), axis=0)
    out = tf.reshape(image, collapsed_shape)  # (sample*N*T, h, w, c)
    out = self.conv1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.conv5(out)
    expanded_shape = tf.concat((tf.shape(image)[:-3], [self.feature_size]), axis=0)
    return tf.reshape(out, expanded_shape)  # (sample, N, T, feature)

class ModelDistributionNetwork(tf.keras.Model):

  def __init__(self,
               observation_spec=(96,96,3),
               action_spec=(3,),
               base_depth=32,
               latent1_size=32,
               latent2_size=256,
               kl_analytic=True,
               model_reward=False,
               model_discount=False,
               decoder_stddev=np.sqrt(0.1, dtype=np.float32),
               reward_stddev=None,
               name=None):
    super(ModelDistributionNetwork, self).__init__(name=name)
    self.observation_spec = observation_spec
    self.action_spec = action_spec
    self.base_depth = base_depth
    self.latent1_size = latent1_size
    self.latent2_size = latent2_size
    self.kl_analytic = kl_analytic
    self.model_reward = model_reward
    self.model_discount = model_discount

    latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag
    latent1_distribution_ctor = MultivariateNormalDiag
    latent2_distribution_ctor = MultivariateNormalDiag

    # p(z_1^1)
    self.latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
    # p(z_1^2 | z_1^1)
    self.latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
    # p(z_{t+1}^1 | z_t^2, a_t)
    self.latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)

    # q(z_1^1 | x_1)
    self.latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    self.latent2_first_posterior = self.latent2_first_prior
    # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
    self.latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
    # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    self.latent2_posterior = self.latent2_prior

    # compresses x_t into a vector
    self.compressor = Compressor(base_depth, 8 * base_depth)
    # p(x_t | z_t^1, z_t^2)
    self.decoder = Decoder(base_depth, scale=decoder_stddev)

    if self.model_reward:
      # p(r_t | z_t^1, z_t^2, a_t, z_{t+1}^1, z_{t+1}^2)
      self.reward_predictor = Normal(8 * base_depth, scale=reward_stddev)

  @property
  def state_size(self):
    return self.latent1_size + self.latent2_size

  def compute_loss(self, images, actions, step_types=None, rewards=None, discounts=None, latent_posterior_samples_and_dists=None):
    sequence_length = actions.shape[1] - 1
    batch_size = actions.shape[0]

    if latent_posterior_samples_and_dists is None:
      latent_posterior_samples_and_dists = self.sample_posterior(images, actions)
    (latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
        latent_posterior_samples_and_dists)


    def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    reset_masks = tf.concat([tf.ones((batch_size,1),dtype=tf.bool),
                             tf.zeros((batch_size,sequence_length),dtype=tf.bool)],axis=1)

    latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, self.latent1_size])
    # inputs is only for inferer batch size
    latent1_first_prior_dists = self.latent1_first_prior(latent1_reset_masks) 
    # these distributions start at t=1 and the inputs are from t-1
    latent1_after_first_prior_dists = self.latent1_prior(
        latent2_posterior_samples[:, :sequence_length], actions[:, :sequence_length])
    latent1_prior_dists = nest_utils.map_distribution_structure(
        functools.partial(where_and_concat, latent1_reset_masks),
        latent1_first_prior_dists,
        latent1_after_first_prior_dists)

    if self.kl_analytic:
      latent1_kl_divergences = tfd.kl_divergence(latent1_posterior_dists, latent1_prior_dists)
    else:
      latent1_kl_divergences = (latent1_posterior_dists.log_prob(latent1_posterior_samples)
                                - latent1_prior_dists.log_prob(latent1_posterior_samples))
    latent1_kl_divergences = tf.reduce_sum(latent1_kl_divergences, axis=1)
    
   # because latent2_posterior == latent2_prior
    latent2_kl_divergences = 0.0

    likelihood_dists = self.decoder(latent1_posterior_samples, latent2_posterior_samples)
    likelihood_log_probs = likelihood_dists.log_prob(images)
    likelihood_log_probs = tf.reduce_sum(likelihood_log_probs, axis=1)
    reconstruction_error = tf.reduce_sum(tf.square(images - likelihood_dists.distribution.loc),
                                         axis=list(range(-len(likelihood_dists.event_shape), 0)))
    reconstruction_error = tf.reduce_sum(reconstruction_error, axis=1)

    # summed over the time dimension
    elbo = likelihood_log_probs - latent1_kl_divergences - latent2_kl_divergences
   
    # average over the batch dimension
    loss = -tf.reduce_mean(elbo)
    return loss

  def sample_posterior(self, images, actions, step_types=None, features=None):
    sequence_length = actions.shape[1]- 1
    actions = actions[:, :sequence_length]

    if features is None:
      features = self.compressor(images)

    # swap batch and time axes
    features = tf.transpose(features, [1, 0, 2])
    actions = tf.transpose(actions, [1, 0, 2])

    latent1_dists = []
    latent1_samples = []
    latent2_dists = []
    latent2_samples = []
    for t in range(sequence_length + 1):
      if t == 0:
        latent1_dist = self.latent1_first_posterior(features[t])
        latent1_sample = latent1_dist.sample()
        latent2_dist = self.latent2_first_posterior(latent1_sample,)
        latent2_sample = latent2_dist.sample()
      else:
        latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
        latent1_sample = latent1_dist.sample()
        latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
        latent2_sample = latent2_dist.sample()

      latent1_dists.append(latent1_dist)
      latent1_samples.append(latent1_sample)
      latent2_dists.append(latent2_dist)
      latent2_samples.append(latent2_sample)

    latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)
    latent1_samples = tf.stack(latent1_samples, axis=1)
    latent2_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent2_dists)
    latent2_samples = tf.stack(latent2_samples, axis=1)

    return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)
