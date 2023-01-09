import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, max_size=100000, observation_shape=(3,), action_shape=(1,)):
        self.mem_size = max_size
        self.mem_cntr = 0
        #------------------------------------------------------------
        self.observation_memory = np.zeros((self.mem_size, *observation_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape),dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.step_type_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.next_observation_memory = np.zeros((self.mem_size, *observation_shape),dtype=np.float32)
	#----------------------------------------------------------------
    def store_transition(self, observation, action, reward, step_type, done, next_observation):
        index = self.mem_cntr % self.mem_size
		#------------------------------------------------------------
        self.observation_memory[index] = observation
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.step_type_memory[index] = step_type
        self.done_memory[index] = done
        self.next_observation_memory[index] = next_observation
		#------------------------------------------------------------
        self.mem_cntr += 1
	#----------------------------------------------------------------
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
		#------------------------------------------------------------
        batch = np.random.choice(max_mem, batch_size, replace=False)
		#------------------------------------------------------------
        observations = tf.convert_to_tensor(self.observation_memory[batch],dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_memory[batch],dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory[batch],dtype=tf.float32)
        step_types = tf.convert_to_tensor(self.step_type_memory[batch],dtype=tf.float32)
        dones = tf.convert_to_tensor(self.done_memory[batch],dtype=tf.float32)
        next_observations = tf.convert_to_tensor(self.next_observation_memory[batch],dtype=tf.float32)
		#------------------------------------------------------------
        return observations, actions, rewards, step_types, dones, next_observations
