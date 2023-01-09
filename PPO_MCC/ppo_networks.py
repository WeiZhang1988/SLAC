import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense 


class ActorNetwork(keras.Model):
    def __init__(self, action_shape=(1,), fc1_dims=128, fc2_dims=64):
        super(ActorNetwork, self).__init__()
        self.action_shape = action_shape[0]
        self.noise = 1e-6

        self.fc1 = Dense(fc1_dims, activation='relu', name='actor_fc1')
        #self.fc2 = Dense(fc2_dims, activation='relu', name='actor_fc2')
        self.mu = Dense(self.action_shape, activation='tanh', name='actor_mu')
        self.sigma = Dense(self.action_shape, activation='sigmoid', name='actor_sigma')

    def call(self, state):
        prob = self.fc1(state)
        #prob = self.fc2(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        #sigma = tf.clip_by_value(sigma,1e-5,2.0)

        return mu, sigma
    
    def cal_log_prob(self, state, action):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        log_prob = probabilities.log_prob(action)
        log_prob = tf.expand_dims(log_prob,axis=-1)

        return log_prob
        
    def sample_normal(self, state, reparameterize=False):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.MultivariateNormalDiag(mu, sigma)
		#------------------------------------------------------------
        if reparameterize:
            probabilities_std = tfp.distributions.MultivariateNormalDiag(tf.zeros_like(mu), tf.ones_like(sigma))
            epsilon = probabilities_std.sample()
            action_origin = epsilon*sigma + mu
        else:
            action_origin = probabilities.sample()
		#------------------------------------------------------------
        action = tf.clip_by_value(action_origin,-1.0,1.0)
        log_prob = probabilities.log_prob(action)
        log_prob = tf.expand_dims(log_prob,axis=-1)
        
        return action, log_prob


class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu', name='value_fc1')
        #self.fc2 = Dense(fc2_dims, activation='relu', name='value_fc2')
        self.value = Dense(1, activation=None, name='value_value')

    def call(self, state):
        x = self.fc1(state)
        #x = self.fc2(x)
        value = self.value(x)

        return value
