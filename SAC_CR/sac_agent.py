import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from sac_networks import ActorNetwork, CriticNetwork

class SacAgent(object):
	def __init__(self, replay_buffer, actor_learning_rate=3e-4, \
			critic_learning_rate=3e-4, alpha_learning_rate=3e-4, \
			batch_size=256, target_update_tau=0.005, target_update_period=10, \
			gamma=0.999, reward_scale_factor=2.0, target_entropy=-3.0):
		#------------------------------------------------------------
		self.replay_buffer = replay_buffer
		self.actor_network = ActorNetwork()
		self.critic_network1 = CriticNetwork()
		self.critic_network2 = CriticNetwork()
		self.target_critic_network1 = CriticNetwork()
		self.target_critic_network2 = CriticNetwork()
		#------------------------------------------------------------
		self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
		self.critic1_optimizer = Adam(learning_rate=critic_learning_rate)
		self.critic2_optimizer = Adam(learning_rate=critic_learning_rate)
		self.alpha_optimizer = Adam(learning_rate=alpha_learning_rate)
		#------------------------------------------------------------
		self.actor_network.compile(optimizer=self.actor_optimizer)
		self.critic_network1.compile(optimizer=self.critic1_optimizer)
		self.critic_network2.compile(optimizer=self.critic2_optimizer)
		#------------------------------------------------------------
		self.batch_size = batch_size
		self.target_update_tau = target_update_tau
		self.target_update_period = target_update_period
		self.gamma = gamma
		self.reward_scale_factor = reward_scale_factor
		#------------------------------------------------------------
		self.update_network_parameters(tau=1.0)
		#------------------------------------------------------------
		self.learning_step = 0
		self.alpha = tf.Variable([1.0])
		self.target_entropy = target_entropy
	#----------------------------------------------------------------
	def choose_action(self, observation):
		obs_tensor = tf.convert_to_tensor([observation],dtype=tf.float32)
		action, _ = self.actor_network.sample_normal(obs_tensor, reparameterize=False)
		action = tf.squeeze(action,axis=0).numpy()
		#------------------------------------------------------------
		return action
	#----------------------------------------------------------------
	def store_transition(self, observation, action, reward, step_type, next_observation):
		self.replay_buffer.store_transition(observation, action, reward, step_type, next_observation)
	#----------------------------------------------------------------
	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.target_update_tau
		#------------------------------------------------------------
		updated_target_critic_weights1 = []
		target_critic_weights1 = self.target_critic_network1.get_weights()
		for i, weight in enumerate(self.critic_network1.get_weights()):
			updated_target_critic_weights1.append(weight * tau + target_critic_weights1[i] * (1.0-tau))
		self.target_critic_network1.set_weights(updated_target_critic_weights1)
		#------------------------------------------------------------
		updated_target_critic_weights2 = []
		target_critic_weights2 = self.target_critic_network2.get_weights()
		for i, weight in enumerate(self.critic_network2.get_weights()):
			updated_target_critic_weights2.append(weight * tau + target_critic_weights2[i] * (1.0-tau))
		self.target_critic_network2.set_weights(updated_target_critic_weights2)
	#----------------------------------------------------------------
	def save_models(self):
		pass
	#----------------------------------------------------------------
	def load_models(self):
		pass
	#----------------------------------------------------------------
	def learn(self):
		if self.replay_buffer.mem_cntr < self.batch_size:
			return
		#------------------------------------------------------------
		observations, actions, rewards, step_typs, next_observations = \
			self.replay_buffer.sample_buffer(self.batch_size)
		#------------------------------------------------------------
		with tf.GradientTape() as tape:
			new_policy_actions, new_policy_log_probs = self.actor_network.sample_normal(observations, reparameterize=True)
			critic1_new_policy = self.critic_network1(observations, new_policy_actions)
			critic2_new_policy = self.critic_network2(observations, new_policy_actions)
			critic_new_policy = tf.math.minimum(critic1_new_policy, critic2_new_policy)
			actor_network_loss = tf.math.reduce_mean(self.alpha * new_policy_log_probs - critic_new_policy)
		actor_network_gradient = tape.gradient(actor_network_loss, self.actor_network.trainable_variables)
		self.actor_network.optimizer.apply_gradients(zip(actor_network_gradient, self.actor_network.trainable_variables))
		#------------------------------------------------------------
		with tf.GradientTape(persistent=True) as tape:
			#???whether the next actions should be new policy or old policy??? I think it should be new policy for now.
			next_new_policy_actions, next_new_policy_log_probs = self.actor_network.sample_normal(next_observations, reparameterize=True)
			td_target1 = self.reward_scale_factor * rewards + self.gamma * self.target_critic_network1(next_observations, next_new_policy_actions)
			td_target2 = self.reward_scale_factor * rewards + self.gamma * self.target_critic_network2(next_observations, next_new_policy_actions)
			critic1_old_policy = self.critic_network1(observations, actions)
			critic2_old_policy = self.critic_network2(observations, actions)
			critic_network1_loss = 0.5 * keras.losses.MSE(critic1_old_policy, td_target1)
			critic_network2_loss = 0.5 * keras.losses.MSE(critic2_old_policy, td_target2)
		critic_network1_gradient = tape.gradient(critic_network1_loss, self.critic_network1.trainable_variables)
		critic_network2_gradient = tape.gradient(critic_network2_loss, self.critic_network2.trainable_variables)
		self.critic_network1.optimizer.apply_gradients(zip(critic_network1_gradient, self.critic_network1.trainable_variables))
		self.critic_network2.optimizer.apply_gradients(zip(critic_network2_gradient, self.critic_network2.trainable_variables))
		#------------------------------------------------------------
		self.learning_step += 1
		#------------------------------------------------------------
		if self.learning_step % self.target_update_period == 0:
			self.update_network_parameters()
		#------------------------------------------------------------
		with tf.GradientTape() as tape:
			_, new_policy_log_probs_alpha = self.actor_network.sample_normal(observations, reparameterize=True)
			alpha_loss = -self.alpha * (new_policy_log_probs_alpha + self.target_entropy) 
		alpha_gradient = tape.gradient(alpha_loss, self.alpha)
