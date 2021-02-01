import pickle
import dill
import numpy as np
from agent.agent import Agent
from collections import defaultdict


class Player(Agent):

    def __init__(self, env_width, env_height, latent_dim=1, learning_rate=0.1, q_table=None):
        super().__init__(env_width, env_height)
        self.current_position = np.array([env_width - 1, 0])
        self.target_position = np.array([env_width - 1, env_height - 1])
        self.learning_rate = learning_rate
        self.q_table = q_table if q_table is not None else defaultdict(
            lambda: np.zeros((len(self.action_space), latent_dim)))

    def learn(self, reward, discount_factor, old_state, new_state, action, z):
        best_next_action = self.select_best_action(new_state, z)
        td_target = reward + discount_factor * self.q_table[new_state][best_next_action, z]
        self.q_table[old_state][action, z] += self.learning_rate * (td_target - self.q_table[old_state][action, z])

    def take_action(self, state, z=None, epsilon=0.1):
        best_action = self.select_best_action(state, z)
        action = self.epsilon_greedy(best_action, epsilon)
        return action

    def select_best_action(self, state, z):
        temp = self.q_table[state]
        # we know the latent policy
        if z is not None:
            temp = temp[:, z]
            return temp.argmax()
        return np.unravel_index(temp.argmax(), temp.shape)[0]

    def reset(self):
        self.current_position = np.array([self.env_width - 1, 0])

    def save(self, path):
        with open(path, "wb") as f:
            dill.dump(self.q_table, f)

    @staticmethod
    def load(path, env_width, env_height):
        with open(path, "rb") as f:
            q_table = dill.load(f)
        player = Player(env_width, env_height, q_table=q_table)
        return player
