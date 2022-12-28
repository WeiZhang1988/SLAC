import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import Compressor,Decoder,ConstantMultivariateNormalDiag,MultivariateNormalDiag,Normal,ModelDistributionNetwork

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
image_batch = np.stack([image_sequence] * batch_size) / 255.0
action_batch = np.stack([action_sequence] * batch_size)
action_batch = tf.cast(action_batch,tf.float32)
observation_spec = (96,96,3)
action_spec = (3,)

model = ModelDistributionNetwork(observation_spec, action_spec)

latent_posterior_samples_and_dists = model.sample_posterior(image_batch,action_batch)
(latent1_posterior_samples, latent2_posterior_samples), (latent1_posterior_dists, latent2_posterior_dists) = (
        latent_posterior_samples_and_dists)
        
loss = model.compute_loss(image_batch,action_batch)

print(f'loss:{loss}')