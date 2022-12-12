import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer

obs = np.random.rand(30,96,96,3)
act = np.random.rand(30,3)
rwd = np.random.rand(30,1)
stp = np.random.rand(30,1)
don = np.random.rand(30,1)
next_obs = np.random.rand(30,96,96,3)

rb = ReplayBuffer()

for i in range(30):
	rb.store_transition(obs[i],act[i],rwd[i],stp[i],don[i],next_obs[i])

sample = rb.sample_buffer(10,5)

print("length: ",len(sample))
print("obs shape: ",tf.shape(sample[0][0]))
print("act shape: ",tf.shape(sample[1][0]))
print("rwd shape: ",tf.shape(sample[2][0]))
print("stp shape: ",tf.shape(sample[3][0]))
print("don shape: ",tf.shape(sample[4][0]))
print("next_obs shape: ",tf.shape(sample[5][0]))
