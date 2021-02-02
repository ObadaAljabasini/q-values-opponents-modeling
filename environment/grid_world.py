import hashlib
import matplotlib.pyplot as plt
from functools import partial
import gym
import numpy as np
from gym_minigrid.envs import Grid, Goal, Lava, Wall, Floor, Ball, Key, Door

from gym_minigrid.window import Window


class GridWorldEnv(gym.Env):

    def __init__(self, player, opponent, width=5, height=5, max_steps=100):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.step_count = 0
        self.player = player
        self.opponent = opponent
        action_space_dim = len(player.action_space)
        self.action_space = gym.spaces.MultiDiscrete([action_space_dim, action_space_dim])
        self.grid: Grid = None
        self.player_obj = Ball(color="green")
        self.opponent_obj = Lava()
        self.wall_type = Wall().__class__.__name__
        self.reset()

    def action_to_move(self, action):
        return self.player.action_to_move[action]

    def step(self, actions):
        self.step_count += 1
        player_action, opponent_action = actions
        player_action, opponent_action = self.action_to_move(player_action), self.action_to_move(opponent_action)
        new_player_position = self.player.current_position + player_action
        new_opponent_position = self.opponent.current_position + opponent_action
        if self.is_allowed_position(new_player_position):
            self.put_object(Floor(), self.player.current_position)
            self.player.current_position = new_player_position
            self.put_object(self.player_obj, new_player_position)
        if self.is_allowed_position(new_opponent_position):
            self.put_object(Floor(), self.opponent.current_position)
            self.opponent.current_position = new_opponent_position
            self.put_object(self.opponent_obj, new_opponent_position)

        obs = self.generate_observation()
        reward = -1
        done = False
        # reached the goal
        if np.array_equal(self.player.current_position, self.player.target_position):
            reward = 100
            done = True
        if self.step_count == self.max_steps:
            done = True
        if np.array_equal(self.player.current_position, self.opponent.current_position):
            reward = -10
        return obs, reward, done

    def reset(self):
        self.step_count = 0
        self.player.reset()
        self.opponent.reset()
        self.generate_grid(self.width, self.height)
        return self.generate_observation()

    def generate_observation(self):
        h = hashlib.sha256()
        image = self.grid.encode()
        h.update(str(image).encode('utf8'))
        hcode = h.hexdigest()[:16]
        return image, hcode

    def is_allowed_position(self, position):
        x, y = position
        if x < 0 or y < 0 or x >= self.height or y >= self.height:
            return False
        obj = self.grid.get(x, y)
        return self.get_type(obj) != "Wall"

    @staticmethod
    def get_type(obj):
        return obj.__class__.__name__

    def put_object(self, obj, position):
        i, j = position
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def render(self, mode='human', tile_size=32):
        image = self.grid.render(tile_size, [1, 1])
        # self.window.show(block=False)
        # image = image.transpose([1, 0, 2])
        # plt.imshow(image, interpolation='bilinear')
        # self.window.show_img(image)
        return image

    def close(self):
        pass
        # if self.window:
        #     self.window.close()

    def generate_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        for i in range(self.height):
            for j in range(self.width):
                self.put_object(Floor(), (i, j))
        # Place a goal square in the bottom-right corner
        # self.put_object(Goal(), (width - 1, height - 1))
        self.put_object(Key(color="yellow"), (width - 1, height - 1))
        f = partial(Wall, color="red")
        self.grid.horz_wall(0, height // 2, obj_type=f)
        # self.grid.set(width // 2, height // 2, Door(color='red', is_open=True, is_locked=False))
        self.grid.set(width // 2, height // 2, Floor())
        self.put_object(self.opponent_obj, self.opponent.current_position)
        self.put_object(self.player_obj, self.player.current_position)

