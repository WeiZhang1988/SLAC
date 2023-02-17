import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, max_size=10000, image_shape=(96,96,3), gnss_shape=(3,), target_shape=(3,), action_shape=(3,)):
        self.mem_size = max_size
        self.mem_cntr = 0
        #------------------------------------------------------------
        self.image_memory = np.zeros((self.mem_size, *image_shape),dtype=np.float32)
        self.gnss_memory = np.zeros((self.mem_size, *gnss_shape),dtype=np.float32)
        self.target_memory = np.zeros((self.mem_size, *target_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape),dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.step_type_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size, 1), dtype=np.float32)
        self.next_image_memory = np.zeros((self.mem_size, *image_shape),dtype=np.float32)
        self.next_gnss_memory = np.zeros((self.mem_size, *gnss_shape),dtype=np.float32)
        self.next_target_memory = np.zeros((self.mem_size, *target_shape),dtype=np.float32)
	#----------------------------------------------------------------
    def store_transition(self, image, gnss, target, action, reward, step_type, done, next_image, next_gnss, next_target):
        index = self.mem_cntr % self.mem_size
		#------------------------------------------------------------
        self.image_memory[index] = image
        self.gnss_memory[index] = gnss
        self.target_memory[index] = target
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.step_type_memory[index] = step_type
        self.done_memory[index] = done
        self.next_image_memory[index] = next_image
        self.next_gnss_memory[index] = next_gnss
        self.next_target_memory[index] = next_target
		#------------------------------------------------------------
        self.mem_cntr += 1
	#----------------------------------------------------------------
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
		#------------------------------------------------------------
        batch = np.random.choice(max_mem, batch_size, replace=False)
		#------------------------------------------------------------
        images = tf.convert_to_tensor(self.image_memory[batch],dtype=tf.float32)
        gnsses = tf.convert_to_tensor(self.gnss_memory[batch],dtype=tf.float32)
        targets = tf.convert_to_tensor(self.target_memory[batch],dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_memory[batch],dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory[batch],dtype=tf.float32)
        step_types = tf.convert_to_tensor(self.step_type_memory[batch],dtype=tf.float32)
        dones = tf.convert_to_tensor(self.done_memory[batch],dtype=tf.float32)
        next_images = tf.convert_to_tensor(self.next_image_memory[batch],dtype=tf.float32)
        next_gnsses = tf.convert_to_tensor(self.next_gnss_memory[batch],dtype=tf.float32)
        next_targets = tf.convert_to_tensor(self.next_target_memory[batch],dtype=tf.float32)
		#------------------------------------------------------------
        return images, gnsses, targets, actions, rewards, step_types, dones, next_images, next_gnsses, next_targets
