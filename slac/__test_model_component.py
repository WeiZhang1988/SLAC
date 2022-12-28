
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Compressor,Decoder,ConstantMultivariateNormalDiag,MultivariateNormalDiag,Normal

import nest_utils

image_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/slac/'
image = np.load(image_path+'figure.npy')
# hyper-parameter
sequence_length = 8
base_depth = 32
latent1_size = 32
latent2_size = 256
batch_size = 64
# input
image_sequence = np.stack([image] * (sequence_length + 1))
action_sequence = np.stack([np.zeros(3)] * (sequence_length + 1))
image_batch = np.stack([image_sequence] * batch_size)
action_batch = np.stack([action_sequence] * batch_size)

# compressor # OK
compressor = Compressor(base_depth,base_depth * 8)
decoder = Decoder(base_depth, scale = 0.3162)

feature_batch = compressor(image_batch)
print('feature batch',feature_batch.shape) # sample × sequence × feature_size
#  
latent1_first_prior_distribution_ctor = ConstantMultivariateNormalDiag 
latent1_distribution_ctor = MultivariateNormalDiag
latent2_distribution_ctor = MultivariateNormalDiag

# p(z_1^1)
latent1_first_prior = latent1_first_prior_distribution_ctor(latent1_size)
# p(z_1^2 | z_1^1)
latent2_first_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)
# p(z_{t+1}^1 | z_t^2, a_t)
latent1_prior = latent1_distribution_ctor(8 * base_depth, latent1_size)
# p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
latent2_prior = latent2_distribution_ctor(8 * base_depth, latent2_size)

# q(z_1^1 | x_1)
latent1_first_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
# q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
latent2_first_posterior = latent2_first_prior
# q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
latent1_posterior = latent1_distribution_ctor(8 * base_depth, latent1_size)
# q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
latent2_posterior = latent2_prior


# # posterior
with tf.name_scope('posterior'):
    # q(z_1^1 | x_1)
    latent1_first_posterior_dists = latent1_first_posterior(feature_batch[:,1,:])
    latent1_first_sample = latent1_first_posterior_dists.sample()
    # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
    latent2_first_posterior_dists = latent2_first_posterior(latent1_first_sample,)
    latent2_first_sample = latent2_first_posterior_dists.sample()
    # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
    latent1_posterior_dists = latent1_posterior(feature_batch[:,1,:],latent2_first_sample,action_batch[:,1,:])
    latent1_sample = latent1_posterior_dists.sample()
    # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    latent2_posterior_dists = latent2_posterior(latent1_sample,latent2_first_sample,action_batch[:,1,:])
    latent2_sample = latent2_posterior_dists.sample()
    print('posterior over')


# prior
with tf.name_scope('porior'):
    # 1. p(z_1^1)
    latent1_first_prior_dists = latent1_first_prior(feature_batch[:,1,:]) # only use the batch_size
    #latent1_first_sample = latent1_first_prior_dists.sample()
    # 2. p(z_1^2 | z_1^1)
    latent2_first_prior_dists = latent2_first_prior(latent1_first_sample,) 
    #latent2_first_sample = latent2_first_prior_dists.sample()
    # 3. p(z_{t+1}^1 | z_t^2, a_t)
    latent1_porior_dists = latent1_prior(latent2_first_sample,action_batch[:,1,:])
    #latent1_sample = latent1_posterior_dists.sample()
    # 4. p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
    latent1_posterior_dists = latent2_prior(latent1_sample,latent2_first_sample,action_batch[:,1,:])
    #latent2_sample = latent1_posterior_dists.sample()
    print('porior over')

latent1_dists = [latent1_porior_dists] * 9


latent1_dists = nest_utils.map_distribution_structure(lambda *x: tf.stack(x, axis=1), *latent1_dists)

def where_and_concat(reset_masks, first_prior_tensors, after_first_prior_tensors):
      after_first_prior_tensors = tf.where(reset_masks[:, 1:], first_prior_tensors[:, 1:], after_first_prior_tensors)
      prior_tensors = tf.concat([first_prior_tensors[:, 0:1], after_first_prior_tensors], axis=1)
      return prior_tensors

    # reset_masks = tf.concat([tf.ones_like(step_types[:, 0:1], dtype=tf.bool),
    #                          tf.equal(step_types[:, 1:], ts.StepType.FIRST)], axis=1)
# 1 0 0 0 0 0 0 0 0
reset_masks = tf.concat([tf.ones((batch_size,1),dtype=tf.bool),
                            tf.zeros((batch_size,sequence_length),dtype=tf.bool)],axis=1)

latent1_reset_masks = tf.tile(reset_masks[:, :, None], [1, 1, latent1_size])
latent1_first_prior_dists = latent1_first_prior(feature_batch[:,:,:])
# latent2 as inputs
latent2_first_sample_batch = tf.tile(latent2_first_sample[:,None,:],[1,8,1])
latent1_after_first_prior_dists = latent1_prior(latent2_first_sample_batch[:,:sequence_length],action_batch[:,:sequence_length])

import functools
latent1_prior_dists = nest_utils.map_distribution_structure(
    functools.partial(where_and_concat, latent1_reset_masks),
    latent1_first_prior_dists,
    latent1_after_first_prior_dists)

decode_image_distribution = decoder(latent1_sample,latent2_sample)
decode_image = decode_image_distribution.mean()





