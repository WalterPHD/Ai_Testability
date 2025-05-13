import gym
from gym import spaces
import numpy as np

class TagEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(TagEnv, self).__init__()
        self.grid_size = 10

        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)

        # Observation: chaser x, y and runner x, y
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.chaser_pos = np.array([0, 0])
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.chaser_pos, self.runner_pos])

    def step(self, action):
        # Move chaser
        if action == 0:
            self.chaser_pos[1] -= 1
        elif action == 1:
            self.chaser_pos[1] += 1
        elif action == 2:
            self.chaser_pos[0] -= 1
        elif action == 3:
            self.chaser_pos[0] += 1

        self.chaser_pos = np.clip(self.chaser_pos, 0, self.grid_size - 1)

        # Move runner randomly (you can later improve this)
        move = np.random.choice(4)
        if move == 0:
            self.runner_pos[1] -= 1
        elif move == 1:
            self.runner_pos[1] += 1
        elif move == 2:
            self.runner_pos[0] -= 1
        elif move == 3:
            self.runner_pos[0] += 1
        self.runner_pos = np.clip(self.runner_pos, 0, self.grid_size - 1)

        # Check if chaser has tagged the runner
        done = np.array_equal(self.chaser_pos, self.runner_pos)

        # Update rewards
        if done:
            chaser_reward = 1.0  # Chaser gets a reward when it tags the runner
            runner_reward = -1.0  # Runner gets a negative reward when it gets tagged
        else:
            chaser_reward = -0.01  # Small penalty to encourage faster tagging
            runner_reward = 0.1  # Small reward to encourage the runner to avoid being tagged

        return self._get_obs(), chaser_reward, done, {"runner_reward": runner_reward}

    def render(self, mode="human"):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        cx, cy = self.chaser_pos
        rx, ry = self.runner_pos
        grid[cy][cx] = "C"
        grid[ry][rx] = "R"
        print("\n".join(" ".join(row) for row in grid))
        print("-" * 20)
