import pandas as pd
import altair as alt
from agent.opponent import Opponent
from agent.player import Player
from environment.grid_world import GridWorldEnv
from array2gif import write_gif


def plot_rewards(rewards, path):
    df = pd.DataFrame({"reward": rewards})
    df["episode"] = df.index + 1
    chart = alt.Chart(df).mark_line().encode(
        x="episode",
        y="reward"
    )
    chart.save(path)


def train_player(env_width, env_height, player, z, nb_episodes, horizon=100):
    opponent = Opponent(env_width, env_height, z)
    env = GridWorldEnv(player, opponent, env_width, env_height, horizon)
    discount_factor = 0.7
    epsilon = 0.9
    episode_rewards = []
    for episode in range(1, nb_episodes + 1):
        print(f"Episode {episode}: training with z = {z}")
        epsilon *= 0.9
        old_state = env.reset()
        done = False
        frames = []
        episode_reward = 0
        while not done:
            player_action = player.take_action(old_state, z)
            opponent_action = opponent.take_action(old_state)
            new_state, reward, done = env.step([player_action, opponent_action])
            player.learn(reward, discount_factor, old_state, new_state, player_action, z)
            old_state = new_state
            episode_reward += reward
            frame = env.render()
            frames.append(frame)
        episode_rewards.append(episode_reward)
        file_name = f"episodes/training/{z}/episode_{episode}.gif"
        write_gif(frames, file_name, 3)
    plot_rewards(episode_rewards, f"plots/{z}.png")


def test_player(env_width, env_height, player, z, horizon=100):
    epsilon = 0
    opponent = Opponent(env_width, env_height, z)
    env = GridWorldEnv(player, opponent, env_width, env_height, horizon)
    print(f"Testing with z = {z}")
    old_state = env.reset()
    done = False
    frames = []
    i = 0
    while not done:
        player_action = player.take_action(old_state, epsilon=epsilon)
        opponent_action = opponent.take_action(old_state, epsilon=epsilon)
        new_state, reward, done = env.step([player_action, opponent_action])
        old_state = new_state
        i += 1
        frame = env.render()
        frames.append(frame)
    file_name = f"episodes/testing/{z}/episode.gif"
    write_gif(frames, file_name, 3)


if __name__ == '__main__':
    env_width, env_height = 5, 5
    # train

    # player = Player(env_width, env_height, 2)
    # nb_episodes = 250
    # for z in [0, 1]:
    #     train_player(env_width, env_height, player, z, nb_episodes)
    # player.save("models/player.pkl")

    # test

    player = Player.load("models/player.pkl", env_width, env_height)
    nb_episodes = 100
    for z in [0, 1]:
        test_player(env_width, env_height, player, z)
