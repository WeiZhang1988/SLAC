import numpy as np
import collections
import random

       
class ReplayBuffer(object):
    def __init__(self,max_len,sequence_len = 8):
        self.buffer = collections.deque(maxlen=max_len)
        self.length = max_len
        self.sequence_len = sequence_len
    
    def process_trajectory(self,trajectory):
        for step in range(len(trajectory) - self.sequence_len):
            sequence = trajectory[step: step + self.sequence_len +1]
            self.buffer.append(sequence)

    def sample(self,batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        observation_sequence_batch, \
        action_sequence_batch, \
        reward_sequence_batch, \
        next_observation_sequence_batch,\
        done_sequence_batch = [], [], [], [], []

        sample_batch = np.array(mini_batch)
        observation_sequence_batch = self.process_shape(sample_batch[:,:,0])
        action_sequence_batch = self.process_shape(sample_batch[:,:,1])
        reward_sequence_batch = self.process_shape(sample_batch[:,:,2])
        next_observation_sequence_batch = self.process_shape(sample_batch[:,:,3])
        done_sequence_batch = self.process_shape(sample_batch[:,:,4])
    
        return  observation_sequence_batch.astype('float32') ,\
                action_sequence_batch.astype('float32') ,\
                reward_sequence_batch.astype('float32') ,\
                next_observation_sequence_batch.astype('float32') ,\
                done_sequence_batch.astype('float32') # sample × n(sequence length)

    def append(self,trajectory):
        if len(trajectory) < (self.sequence_len+1):
            return
        self.process_trajectory(trajectory)
    
    def __len__(self):
        return len(self.buffer)

    def process_shape(self,data):
        # data sample × sequence ×（96，96，3）
        shape = data.shape
        sample = []
        for sample_idx in range(shape[0]):
            sequence = []
            for sequence_idx in range(shape[1]):
                sequence.append(data[sample_idx,sequence_idx])
            sample.append(sequence)
        
        return np.array(sample)