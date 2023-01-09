import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, value_network, gamma=0.99, gae_lambda=0.95, max_size=100000, observation_shape=(2,), action_shape=(1,)):
        self.value_network = value_network
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        #------------------------------------------------------------
        self.mem_size = max_size
        self.mem_cntr = 0
        #------------------------------------------------------------
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.observation_memory = np.zeros((self.mem_size, *observation_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape),dtype=np.float32)
        self.log_prob_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.step_type_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.next_observation_memory = np.zeros((self.mem_size, *observation_shape),dtype=np.float32)
        self.all_observation_memory = np.zeros((self.mem_size+1, *observation_shape),dtype=np.float32)
	#----------------------------------------------------------------
    def store_transition(self, observation, action, log_prob, reward, step_type, done, next_observation):
        index = self.mem_cntr % self.mem_size
		#------------------------------------------------------------
        self.observation_memory[index] = observation
        self.action_memory[index] = action
        self.log_prob_memory[index] = log_prob
        self.reward_memory[index] = reward
        self.step_type_memory[index] = step_type
        self.done_memory[index] = done
        self.next_observation_memory[index] = next_observation
        self.all_observation_memory[index] = observation
        self.all_observation_memory[index+1] = next_observation
		#------------------------------------------------------------
        self.mem_cntr += 1
	#----------------------------------------------------------------
    def cal_targ_agv(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        all_states = tf.convert_to_tensor(self.all_observation_memory[:max_mem],dtype=tf.float32)
        all_values = self.value_network(all_states).numpy()
        values = all_values
        next_values = all_values
        
        gae = 0
        advantage_list = []
        for step in reversed(range(max_mem)):
            delta = self.reward_memory[step] + self.gamma * next_values[step] * (1.0-self.done_memory[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0-self.done_memory[step]) * gae
            # prepend to get correct order back
            advantage_list.insert(0, gae)
        advantages = np.array(advantage_list)
        self.target_memory = advantages + values
        self.advantage_memory = (advantages - np.mean(advantages))/(np.std(advantages) + 1e-10)
	#----------------------------------------------------------------
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
		#------------------------------------------------------------
        batch = np.random.choice(max_mem, batch_size, replace=False)
		#------------------------------------------------------------
        observations = tf.convert_to_tensor(self.observation_memory[batch],dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_memory[batch],dtype=tf.float32)
        log_probs = tf.convert_to_tensor(self.log_prob_memory[batch],dtype=tf.float32)
        targets = tf.convert_to_tensor(self.target_memory[batch],dtype=tf.float32)
        advantages = tf.convert_to_tensor(self.advantage_memory[batch],dtype=tf.float32)
		#------------------------------------------------------------
        return observations, actions, log_probs, targets, advantages
    #----------------------------------------------------------------
    def clear_buffer(self):
        self.mem_cntr = 0
        #------------------------------------------------------------
        self.observation_memory = np.zeros((self.mem_size, *self.observation_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *self.action_shape),dtype=np.float32)
        self.log_prob_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.step_type_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.next_observation_memory = np.zeros((self.mem_size, *self.observation_shape),dtype=np.float32)
        self.all_observation_memory = np.zeros((self.mem_size+1, *self.observation_shape),dtype=np.float32)
