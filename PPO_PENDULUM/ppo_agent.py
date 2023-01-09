import numpy as np
import copy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from replay_buffer import ReplayBuffer
from ppo_networks import ActorNetwork, ValueNetwork
import os

class Agent:
    def __init__(self, gamma=0.99, lr_ac=1e-5, lr_v=2e-5, \
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, \
                 en_coef=0.01, n_epochs=10, chkpt_dir='tmp/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir
        self.actor_network = ActorNetwork()
        self.actor_network.compile(optimizer=Adam(learning_rate=lr_ac))
        self.value_network = ValueNetwork()
        self.value_network.compile(optimizer=Adam(learning_rate=lr_v))
        self.replay_buffer = ReplayBuffer(value_network=self.value_network)
        self.en_coef = tf.constant(en_coef)

    def store_transition(self, observation, action, log_prob, reward, step_type, done, next_observation):
        self.replay_buffer.store_transition(observation, action, log_prob, reward, step_type, done, next_observation)

    def save_models(self):
        print('... saving models ...')
        self.actor_network.save_weights(self.chkpt_dir + 'actor')
        self.value_network.save_weights(self.chkpt_dir + 'value')

    def load_models(self):
        print('... loading models ...')
        self.actor_network.load_weights(self.chkpt_dir + 'actor')
        self.value_network.load_weights(self.chkpt_dir + 'value')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        action, log_prob = self.actor_network.sample_normal(state)
        action = tf.squeeze(action,axis=0).numpy()
        log_prob = tf.squeeze(log_prob,axis=0).numpy()
        return action, log_prob
        
    def choose_action_deterministic(self, observation):
        state = tf.convert_to_tensor([observation])
        action, _ = self.actor_network(state)
        action = tf.squeeze(action,axis=0).numpy()
        return action

    def learn(self):
        self.replay_buffer.cal_targ_agv()
        for _ in range(self.n_epochs):
            states, actions, old_probs, targets, advantages = self.replay_buffer.sample_buffer(self.batch_size)
                
            with tf.GradientTape() as tape:
                new_probs = self.actor_network.cal_log_prob(states,actions)
                prob_ratio = tf.exp(new_probs - old_probs)
                weighted_probs = advantages * prob_ratio
                clipped_probs = tf.clip_by_value(prob_ratio,1.0-self.policy_clip,1.0+self.policy_clip)
                weighted_clipped_probs = clipped_probs * advantages
                actor_loss = tf.minimum(weighted_probs,weighted_clipped_probs)
                actor_loss = -tf.reduce_mean(actor_loss - self.en_coef * new_probs)
                    
            actor_grads = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_network.optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
                
            with tf.GradientTape() as tape:
                critic_values = self.value_network(states)
                critic_loss = 0.5 * tf.reduce_mean(tf.square(targets - critic_values))
            critic_grads = tape.gradient(critic_loss, self.value_network.trainable_variables)
            self.value_network.optimizer.apply_gradients(zip(critic_grads, self.value_network.trainable_variables))

        self.replay_buffer.clear_buffer()
