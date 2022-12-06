import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import tensorflow_probability as tfp
import gym

EPSILON = 1e-6 # log 0 保护
LOG_STD_MAX = 1
LOG_STD_MIN = -10
MAX_ACTION = 2

class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

    def step(self, action):
        action = self._action(action)
        return self.env.step(action)

class Critic(Model):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(256,activation='relu') 
        self.f3 = Dense(128,activation='relu') 
        self.f4 = Dense(1,activation=None) 

    def call(self,observation,action):
        x = self.f1(tf.concat([observation, action], axis=1))
        x = self.f2(x)
        x = self.f3(x)
        q = self.f4(x)
        return q

class Actor(Model):
    def __init__(self,action_dim) -> None:
        super().__init__()
        
        self.f1 = Dense(512,activation='relu')
        self.f2 = Dense(256,activation='relu')
        self.f3 = Dense(128,activation='relu')
        self.mu_f = Dense(action_dim,activation=None)
        self.log_sigma_f = Dense(action_dim,activation=None)

    def call(self,observation):
        x = self.f1(observation)
        x = self.f2(x)
        x = self.f3(x)
        mu = self.mu_f(x)
        log_sigma = self.log_sigma_f(x)
        log_sigma = tf.clip_by_value(log_sigma,LOG_STD_MIN,LOG_STD_MAX)
        sigma = tf.math.exp(log_sigma)
        return mu,sigma
    
    def sample_normal(self,observation):
        mu,sigma = self.call(observation)
        normal = tfp.distributions.Normal(mu, sigma) 
        u      = normal.sample()
        
        action = tf.tanh(u) 
        log_pi = normal.log_prob(u)
        log_pi = log_pi - tf.math.log((1-action**2 + EPSILON))
        log_pi = tf.reduce_sum(log_pi,axis=1,keepdims=True)
        return action,log_pi
