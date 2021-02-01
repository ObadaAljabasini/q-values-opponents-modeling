from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    def __init__(self, env_width, env_height):
        self.env_width = env_width
        self.env_height = env_height
        self.current_position = None
        self.target_position = None
        self.action_space = range(4)
        self.action_to_move = {0: np.array([0, -1]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([1, 0])}
        self.action_mapping = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}

    @abstractmethod
    def take_action(self, observation, epsilon=0.1):
        pass

    def epsilon_greedy(self, action, epsilon):
        p = np.random.random()
        if p < (1 - epsilon):
            return action
        else:
            return np.random.choice(self.action_space, 1)[0]
