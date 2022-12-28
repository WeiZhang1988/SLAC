from actor import Actor
from critic import Critic
from model import ModelDistributionNetwork,Compressor
import numpy as np
import tensorflow as tf
import copy
import pickle
import collections
import common as slac_common

save_path = '/home/ubuntu-1/Learning/ReinforcementLearning/demo/slac/checkpoint/slac'

class Agent(object):
    def __init__(
        self,
        action_dim=3,
        sequence_length=8
        ):
        
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor = Actor(action_dim)

        self.model = ModelDistributionNetwork()

        self.lr = tf.constant(3e-4,dtype=tf.float32)
        self.gamma = tf.constant(0.99,dtype=tf.float32)
        self.tau = tf.constant(0.05,dtype=tf.float32)
        self.reward_scale = tf.constant(1.0,dtype=tf.float32)

        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.observation_sequence=collections.deque(maxlen=sequence_length)
        self.action_sequence = collections.deque(maxlen=sequence_length)
        self.init_feature_sequence()

        self.log_alpha = tf.Variable(0.0,dtype=tf.float32)
        self.alpha = tf.Variable(1.0,dtype=tf.float32)
        self.target_entropy = -tf.constant(action_dim, dtype=tf.float32)

        self.critic1_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.critic2_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.alpha_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.model_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
    
    def init_feature_sequence(self):
        obs = np.zeros((64,64,3))
        action = np.zeros((self.action_dim,))
        for _ in range(self.sequence_length):
            self.observation_sequence.append(obs)
            self.action_sequence.append(action)

    def sync_model(self):
        # updating critic network
        target_weights = self.target_critic1.trainable_variables
        weights = self.critic1.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

        target_weights = self.target_critic2.trainable_variables
        weights = self.critic2.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
        
    def predict(self,observation):

        observation = np.array([observation])
        feature = self.model.compressor(observation)
        action, _ = self.actor.sample_normal(feature)

        return action[0]

    def learn(self,sequence_batch):

        (observations, actions, rewards, next_observations, dones) = (sequence_batch)

        # features
        features_ = self.model.compressor(observations)
        features, next_features = tf.unstack(features_[:,-2:],axis=1)

        # latents
        latent_samples_and_dists = self.model.sample_posterior(
            observations, actions, step_types=None, features=None)
        latents_, _ = latent_samples_and_dists
        latents_ = tf.concat(latents_,axis=-1) # concat (latent1,latent2)
        latents, next_latents = tf.unstack(latents_[:, -2:], axis=1)

        # # sequence action feature
        # sequence_action = actions[:,:-1]
        # sequence_feature = features[:,:-1]
        # sequence_action_feature = tf.concat([sequence_feature,sequence_action],axis=-1)

        # sequence_action = actions[:,1:]
        # sequence_feature = features[:,1:]
        # next_sequence_action_feature = tf.concat([sequence_feature,sequence_action],axis=-1)
        

        with tf.GradientTape(persistent=True) as tape:

            # critic loss
            next_actions,next_log_pis = self.actor.sample_normal(next_features)
            
            q1_next_predict = self.target_critic1(next_latents,next_actions)
            q2_next_predict = self.target_critic2(next_latents,next_actions)
            q_next_predict = tf.math.minimum(q1_next_predict,q2_next_predict)

            soft_q_target = q_next_predict - self.alpha * next_log_pis
            q_target = rewards + (1.0-dones) * self.gamma * soft_q_target
            q1_predict = self.critic1(latents,actions[:,-2,:])
            loss_critic1 = tf.keras.losses.MSE(q_target,q1_predict)
            q2_predict = self.critic2(latents,actions[:,-2,:])
            loss_critic2 = tf.keras.losses.MSE(q_target,q2_predict)

        grads_critic1 = tape.gradient(
            loss_critic1,self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(
            zip(grads_critic1, self.critic1.trainable_variables))
        grads_critic2 = tape.gradient(
            loss_critic2,self.critic2.trainable_variables)
        self.critic1_optimizer.apply_gradients(
            zip(grads_critic2, self.critic1.trainable_variables))

        # actor
        with tf.GradientTape() as tape:
            new_actions,log_pis = self.actor.sample_normal(features)
            q1_predict = self.critic1(latents,new_actions)
            q2_predict = self.critic2(latents,new_actions)
            q_predict  = tf.math.minimum(q1_predict,q2_predict)
            loss_actor = self.alpha*log_pis - q_predict
        grads_actor  = tape.gradient(
            loss_actor,self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads_actor,self.actor.trainable_variables))

        # alpha
        with tf.GradientTape() as tape:
            _, log_pis = self.actor.sample_normal(features)
            loss_alpha  = -1.0*(
                tf.math.exp(self.log_alpha)*(log_pis + self.target_entropy))
            loss_alpha = tf.nn.compute_average_loss(loss_alpha) 
        variables = [self.log_alpha]
        grads_alpha = tape.gradient(loss_alpha,variables)
        self.alpha_optimizer.apply_gradients(
            zip(grads_alpha, variables)
        )
        self.alpha = tf.math.exp(self.log_alpha)

        # model
        with tf.GradientTape() as tape:
            loss_model = self.model.compute_loss(
                images=observations,
                actions=actions)
        grads_model = tape.gradient(
            loss_model,self.model.trainable_variables)
        self.model_optimizer.apply_gradients(
            zip(grads_model,self.model.trainable_variables))

        self.sync_model()

    def save_models(self):
        self.critic1.save_weights(save_path + '-critic1.ckpt')
        self.critic2.save_weights(save_path + '-critic2.ckpt')
        self.actor.save_weights(save_path + '-actor.ckpt')
        self.model.save_weights(save_path + '-model.ckpt')
        data = {'alpha':self.alpha,'log_alpha':self.log_alpha}
        with open(save_path+'-alpha.pickle','wb') as f:
            pickle.dump(data,f)
        

    def load_models(self):
        self.critic1.load_weights(save_path + '-critic1.ckpt')
        self.critic2.load_weights(save_path + '-critic2.ckpt')
        self.actor.load_weights(save_path + '-actor.ckpt')
        self.model.load_weights(save_path + '-model.ckpt')
        with open(save_path+'-alpha.pickle','rb') as f:
            data = pickle.load(f)
            self.alpha = data['alpha']
            self.log_alpha = data['log_alpha']
