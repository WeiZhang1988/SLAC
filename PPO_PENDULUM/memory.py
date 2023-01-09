import numpy as np

class PPOMemory:
    def __init__(self, batch_size=100):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.batch_size = batch_size
    
    def generate_batches(self):
        states_num = len(self.states)
        batch_start = np.arange(0, states_num, self.batch_size)
        indices = np.arange(states_num, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches
        
    def get_array(self):
        return np.array(self.states).astype("float32"),\
            np.array(self.actions).astype("float32"),\
            np.array(self.log_probs).astype("float32"),\
            np.array(self.rewards).astype("float32"),\
            np.array(self.next_states).astype("float32"),\
            np.array(self.dones).astype("float32")
            

    def store_memory(self, state, action, log_prob, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
