import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
from slac_networks import ModelNetwork
import nest_utils
import tensorflow_probability as tfp
import functools

obs = tf.convert_to_tensor(np.random.rand(2,5,96,96,3),dtype=tf.float32)
act = tf.convert_to_tensor(np.random.rand(2,5,3),dtype=tf.float32)
rwd = tf.convert_to_tensor(np.random.rand(2,5,1),dtype=tf.float32)
stp = tf.convert_to_tensor(np.random.rand(2,5,1),dtype=tf.float32)
don = tf.convert_to_tensor(np.random.rand(2,5,1),dtype=tf.float32)
next_obs = tf.convert_to_tensor(np.random.rand(2,5,96,96,3),dtype=tf.float32)
"""
print("---stp---",stp)
print("---stp---",stp[:, 0:1])
reset_mask = tf.concat([tf.ones_like(stp[:, 0:1],dtype=tf.bool),tf.equal(stp[:,1:],0)],axis=1)
print("---reset_mask---",reset_mask)
print(reset_mask[:, :, None])
"""

clip_obs = obs[:,:,16:80, 16:80,:]
print(tf.shape(clip_obs))

#m=ModelNetwork()
#print(m.compute_loss(obs,act,stp))

"""
mask=tf.equal(tf.transpose(tf.squeeze(stp,axis=-1))[0],0)
mask=tf.convert_to_tensor([[True],[False]])
print("---mask---",mask)

p1 = tfp.distributions.MultivariateNormalDiag(tf.convert_to_tensor([[0.0, -1.0, 1.0],[0.0, -1.0, 1.0]]), tf.convert_to_tensor([[1.0,1.0,1.0],[1.0,1.0,1.0]]))
print("---p1---",p1)
p2 = tfp.distributions.MultivariateNormalDiag(tf.convert_to_tensor([[0.0, -1.0, 1.0],[0.0, -1.0, 1.0]]), tf.convert_to_tensor([[1.0,1.0,1.0],[1.0,1.0,1.0]]))
print("---p2---",p2)
new_p = nest_utils.map_distribution_structure(functools.partial(tf.where, mask), p1, p2)
print("new_p",new_p)
"""

