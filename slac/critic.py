import tensorflow as tf
import numpy as np

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense
from model import Compressor
import tensorflow_probability as tfp

tfd = tfp.distributions

class Critic(Model):
    def __init__(self) -> None:
        super().__init__()
        self.o1 = Dense(128,activation='relu')
        self.a1 = Dense(128,activation='relu')

        self.f1 = Dense(128,activation='relu') 
        self.f2 = Dense(128,activation='relu') 
        self.f3 = Dense(1,activation=None) 

    def call(self,latent,actions):

        o = self.o1(latent)
        a = self.a1(actions)

        x = self.f1(tf.concat([o, a], axis=1))
        x = self.f2(x)
        q = self.f3(x)

        return q