from agent.agent import Agent
import numpy as np


class Opponent(Agent):

    def __init__(self, env_width, env_height, z=None):
        super().__init__(env_width, env_height)
        self.current_position = np.array([env_width - 2, env_height - 1])
        self.z = z if z is not None else np.random.binomial(size=1, n=1, p=0.5)[0]
        self.target_position = np.array([0, 0]) if self.z == 0 else np.array([0, env_height - 1])

    def take_action(self, observation, epsilon=0.1):
        if self.current_position[0] > self.env_height // 2:
            action = self.action_mapping['Up']
        elif self.current_position[1] > 0 and self.target_position[1] == 0:
            action = self.action_mapping['Left']
        else:
            action = self.action_mapping['Up']
        # action = self.epsilon_greedy(action, epsilon)
        return action

    def reset(self):
        self.current_position = np.array([self.env_width - 2, self.env_height - 1])
