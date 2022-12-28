import tensorflow as tf
import numpy as np

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D,  Activation, MaxPool2D, Dropout, Flatten, Dense
from model import Compressor
import tensorflow_probability as tfp

tfd = tfp.distributions

class Actor(Model):
    def __init__(self,action_dim=3) -> None:
        super().__init__()

        self.flatten = Flatten()
        self.s1 = Dense(128,activation='relu')

        self.f1 = Dense(128,activation='relu') 
        self.f2 = Dense(128,activation='relu') 
        self.f3 = Dense(action_dim,activation=None)
        self.f4 = Dense(action_dim,activation=None)

    def call(self,sequence_action_feature):
        
        x = self.flatten(sequence_action_feature)
        x = self.s1(x)
        x = self.f1(x)
        x = self.f2(x)
        mu = self.f3(x)
        log_sigma = self.f4(x)
        log_sigma = tf.clip_by_value(log_sigma,-20,2)
        sigma = tf.math.exp(log_sigma)
        return mu,sigma
    
    def sample_normal(self,sequence_action_feature):
        mu,sigma = self.call(sequence_action_feature)
         
        # 重参数
        normal_ = tfp.distributions.Normal(tf.zeros(mu.shape),tf.ones(sigma.shape))
        e = normal_.sample()
        log_pi = normal_.log_prob(e)
        action = tf.tanh(mu + e * sigma)
        
        log_pi = log_pi - tf.math.log((1.0-action**2 + 1e-6))
        log_pi = tf.reduce_sum(log_pi,axis=1,keepdims=True)
        return action,log_pi