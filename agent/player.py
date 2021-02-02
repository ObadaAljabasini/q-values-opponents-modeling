from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import dill
import numpy as np

from agent.agent import Agent


class Player(Agent):

    def __init__(self, env_width, env_height, latent_dim=1, learning_rate=0.1, q_table=None, policy_classifier=None):
        super().__init__(env_width, env_height)
        self.current_position = np.array([env_width - 1, 0])
        self.target_position = np.array([env_width - 1, env_height - 1])
        self.learning_rate = learning_rate
        self.data = []
        self.policy_classifier = policy_classifier if policy_classifier is not None else self.create_policy_classifier()
        self.q_table = q_table if q_table is not None else self.create_q_table(latent_dim)

    def learn(self, reward, discount_factor, old_state, new_state, action, z):
        best_next_action = self.select_best_action(new_state, None, z)
        td_target = reward + discount_factor * self.q_table[new_state][best_next_action, z]
        self.q_table[old_state][action, z] += self.learning_rate * (td_target - self.q_table[old_state][action, z])

    def take_action(self, state, image=None, z=None, epsilon=0.1):
        best_action = self.select_best_action(state, image, z)
        action = self.epsilon_greedy(best_action, epsilon)
        return action

    def update_data(self, imgs):
        self.data.extend(imgs)

    def create_policy_classifier(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=50, fit_intercept=True, penalty='l1', solver='saga', tol=0.1))
        ])
        return pipeline

    def create_q_table(self, latent_dim):
        return defaultdict(lambda: np.zeros((len(self.action_space), latent_dim)))

    def train_classifier(self):
        X, y = zip(*self.data)
        # list of array to 2d array
        X = np.stack(X, axis=0)
        y = np.array(y)
        X, y = shuffle(X, y)
        self.policy_classifier.fit(X, y)

    def select_best_action(self, state, image, z):
        temp = self.q_table[state]
        # we know the latent policy
        if z is not None:
            temp = temp[:, z]
            action = temp.argmax()
            return action
        weights = self.policy_classifier.predict_proba(image.reshape(1, -1))
        weights = weights[0]
        temp = temp * weights
        temp = temp.sum(axis=1)
        action = temp.argmax()
        return action
        # return np.unravel_index(temp.argmax(), temp.shape)[0]

    def reset(self):
        self.current_position = np.array([self.env_width - 1, 0])

    def save(self, path):
        with open(path, "wb") as f:
            data = {
                'q_table': self.q_table,
                'policy_classifier': self.policy_classifier
            }
            dill.dump(data, f)

    @staticmethod
    def load(path, env_width, env_height):
        with open(path, "rb") as f:
            data = dill.load(f)
            q_table = data["q_table"]
            policy_classifier = data["policy_classifier"]
        player = Player(env_width, env_height, q_table=q_table, policy_classifier=policy_classifier)
        return player
