import gym
import numpy as np
import collections

class CarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.reward_deque = collections.deque(maxlen=200) #
        self.low = self.action_space.low
        self.high = self.action_space.high
        self.action_repeat = 4

    def _action(self, action):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """
        action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        action = np.clip(action, self.low, self.high)
        
        return action

    def step(self, action):
        action = self._action(action)
        total_reward = 0
        done = False
        truncated = False
        for _ in range(self.action_repeat):
            next_observation, reward, terminated, truncated, info  = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        observation = next_observation[:84,:84,:]
        observation = tf.image.resize(observation,(64,64)).numpy() / 255.0
        done = terminated or truncated
        return observation, total_reward, done, info

    def reset(self):
        self.reward_deque.clear()
        observation,info = self.env.reset()
        observation = observation[:84,:84,:]
        observation = tf.image.resize(observation,(64,64)).numpy() / 255.0
        return observation,info
    
import tensorflow as tf
class RenderWrapper(gym.Wrapper):
    def __init__(self,env,action_repeat=4):
        super().__init__(env)
        self._env = env
        self._action_repeat = action_repeat

    def step(self,action):
        for _ in range(self._action_repeat):
            _, reward, terminated, truncated, info = self._env.step(action)
        observation = self._env.render()
        observation = tf.image.resize(observation,(64,64)).numpy() / 255.0
        done = terminated or truncated
        return observation, reward, done, info
    
    def reset(self):
        _, info = super().reset()
        observation = self._env.render()
        observation = tf.image.resize(observation,(64,64)).numpy() / 255.0
        return observation,info

