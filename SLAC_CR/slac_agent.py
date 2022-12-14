import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pickle
from tensorflow.keras.optimizers import Adam
from replay_buffer import ReplayBuffer
from slac_networks import ActorNetwork, CriticNetwork, ModelNetwork

class SlacAgent(object):
	def __init__(self, actor_learning_rate=3e-4,critic_learning_rate=3e-4, \
			alpha_learning_rate=3e-4, model_learning_rate=3e-4,\
			batch_size=256, target_update_tau=0.005, target_update_period=5, \
			gamma=0.995, reward_scale_factor=2.0, target_entropy=-3.0):
		#------------------------------------------------------------
		self.replay_buffer = ReplayBuffer()
		self.actor_network = ActorNetwork(name="actor")
		self.critic_network1 = CriticNetwork(name="critic1")
		self.critic_network2 = CriticNetwork(name="critic2")
		self.target_critic_network1 = CriticNetwork(name="target_critic1")
		self.target_critic_network2 = CriticNetwork(name="target_critic2")
		self.model_network = ModelNetwork(name="model")
		#------------------------------------------------------------
		self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
		self.critic1_optimizer = Adam(learning_rate=critic_learning_rate)
		self.critic2_optimizer = Adam(learning_rate=critic_learning_rate)
		self.alpha_optimizer = Adam(learning_rate=alpha_learning_rate)
		self.model_optimizer = Adam(learning_rate=model_learning_rate)
		#------------------------------------------------------------
		self.actor_network.compile(optimizer=self.actor_optimizer)
		self.critic_network1.compile(optimizer=self.critic1_optimizer)
		self.critic_network2.compile(optimizer=self.critic2_optimizer)
		self.model_network.compile(optimizer=self.model_optimizer)
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
		self.log_alpha = tf.Variable(0.0,dtype=tf.float32)
		self.target_entropy = tf.constant(target_entropy,dtype=tf.float32)
	#----------------------------------------------------------------
	def choose_action(self, observation):
		obs_tensor = tf.convert_to_tensor([observation],dtype=tf.float32)
		feature = self.model_network.compressor(obs_tensor)
		action, _ = self.actor_network.sample_normal(feature)
		action = tf.squeeze(action,axis=0).numpy()
		#------------------------------------------------------------
		return action
	#----------------------------------------------------------------
	def choose_action_deterministic(self, observation):
		obs_tensor = tf.convert_to_tensor([observation],dtype=tf.float32)
		feature = self.model_network.compressor(obs_tensor)
		action, _ = self.actor_network(feature)
		action = tf.squeeze(action,axis=0).numpy()
		#------------------------------------------------------------
		return action
	#----------------------------------------------------------------
	def store_transition(self, observation, action, reward, step_type, done, next_observation):
		self.replay_buffer.store_transition(observation, action, reward, step_type, done, next_observation)
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
		print('... saving models ...')
		self.actor_network.save_weights(self.actor_network.checkpoint_file)
		self.critic_network1.save_weights(self.critic_network1.checkpoint_file)
		self.critic_network2.save_weights(self.critic_network2.checkpoint_file)
		self.target_critic_network1.save_weights(self.target_critic_network1.checkpoint_file)
		self.target_critic_network2.save_weights(self.target_critic_network2.checkpoint_file)
		self.model_network.save_weights(self.model_network.checkpoint_file)
		with open('tmp/slac/log_alpha.pickle', 'wb') as f:
			pickle.dump(self.log_alpha, f)
	#----------------------------------------------------------------
	def load_models(self):
		print('... loading models ...')
		self.actor_network.load_weights(self.actor_network.checkpoint_file)
		self.critic_network1.load_weights(self.critic_network1.checkpoint_file)
		self.critic_network2.load_weights(self.critic_network2.checkpoint_file)
		self.target_critic_network1.load_weights(self.target_critic_network1.checkpoint_file)
		self.target_critic_network2.load_weights(self.target_critic_network2.checkpoint_file)
		self.model_network.load_weights(self.model_network.checkpoint_file)
		with open('tmp/slac/log_alpha.pickle', 'rb') as f:
			self.log_alpha = pickle.load(f)
	#----------------------------------------------------------------
	def learn(self):
		if self.replay_buffer.mem_cntr < self.batch_size:
			return
		#------------------------------------------------------------
		observations_seq, actions_seq, rewards_seq, step_types_seq, dones_seq, next_observations_seq = \
			self.replay_buffer.sample_buffer(self.batch_size)
		observations, actions, rewards, step_types, dones, next_observations = \
		observations_seq[-1], actions_seq[-1], rewards_seq[-1], step_types_seq[-1], dones_seq[-1], next_observations_seq[-1]
		features_stop = tf.stop_gradient(self.model_network.compressor(observations))
		next_features_stop = tf.stop_gradient(self.model_network.compressor(next_observations))
		#------------------------------------------------------------
		with tf.GradientTape(persistent=True) as tape:
			next_new_policy_actions, next_new_policy_log_probs = self.actor_network.sample_normal(next_features_stop)
			target_critic1 = self.target_critic_network1(next_features_stop, next_new_policy_actions)
			target_critic2 = self.target_critic_network2(next_features_stop, next_new_policy_actions)
			target_critic = tf.minimum(target_critic1, target_critic2)
			td_target = self.reward_scale_factor * rewards + (1.0 - tf.cast(dones, dtype=tf.float32)) * self.gamma * (target_critic - tf.exp(self.log_alpha) * next_new_policy_log_probs)
			critic1_old_policy = self.critic_network1(features_stop, actions)
			critic2_old_policy = self.critic_network2(features_stop, actions)
			critic_network1_loss = 0.5 * keras.losses.MSE(critic1_old_policy, td_target)
			critic_network2_loss = 0.5 * keras.losses.MSE(critic2_old_policy, td_target)
		critic_network1_gradient = tape.gradient(critic_network1_loss, self.critic_network1.trainable_variables)
		critic_network2_gradient = tape.gradient(critic_network2_loss, self.critic_network2.trainable_variables)
		self.critic_network1.optimizer.apply_gradients(zip(critic_network1_gradient, self.critic_network1.trainable_variables))
		self.critic_network2.optimizer.apply_gradients(zip(critic_network2_gradient, self.critic_network2.trainable_variables))
		#------------------------------------------------------------
		with tf.GradientTape() as tape:
			new_policy_actions, new_policy_log_probs = self.actor_network.sample_normal(features_stop)
			critic1_new_policy = self.critic_network1(features_stop, new_policy_actions)
			critic2_new_policy = self.critic_network2(features_stop, new_policy_actions)
			critic_new_policy = tf.math.minimum(critic1_new_policy, critic2_new_policy)
			actor_network_loss = tf.math.reduce_mean(tf.exp(self.log_alpha) * new_policy_log_probs - critic_new_policy)
		actor_network_gradient = tape.gradient(actor_network_loss, self.actor_network.trainable_variables)
		self.actor_network.optimizer.apply_gradients(zip(actor_network_gradient, self.actor_network.trainable_variables))
		#------------------------------------------------------------
		with tf.GradientTape() as tape:
			new_policy_actions, new_policy_log_probs = self.actor_network.sample_normal(features_stop)
			alpha_loss = -tf.exp(self.log_alpha) * (new_policy_log_probs + self.target_entropy) 
			alpha_loss = tf.nn.compute_average_loss(alpha_loss) 
		alpha_gradient = tape.gradient(alpha_loss, [self.log_alpha])
		self.alpha_optimizer.apply_gradients(zip(alpha_gradient,[self.log_alpha]))
		#------------------------------------------------------------
		with tf.GradientTape() as tape:
			model_network_loss = self.model_network.compute_loss(observations_seq, actions_seq, step_types_seq)
		model_network_gradient = tape.gradient(model_network_loss, self.model_network.trainable_variables)
		self.model_network.optimizer.apply_gradients(zip(model_network_gradient, self.model_network.trainable_variables))
		#------------------------------------------------------------
		if self.learning_step % self.target_update_period == 0:
			self.update_network_parameters()
		#------------------------------------------------------------
		self.learning_step += 1
