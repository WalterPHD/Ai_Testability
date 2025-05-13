from stable_baselines3 import PPO
from tag_env import TagEnv
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import animation
import os

# Load the trained model
env = TagEnv()
model = PPO.load(r"D:\VS Stuff\AI\Tag\train\tag_model")

# Storage for frames
frames = []

# Reset the environment
obs = env.reset()
done = False
steps = 0
max_steps = 100

# Run the simulation and collect frames
while not done and steps < max_steps:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    # Create frame grid
    grid = np.zeros((env.grid_size, env.grid_size))
    cx, cy = env.chaser_pos
    rx, ry = env.runner_pos
    grid[cy][cx] = 1  # Chaser
    grid[ry][rx] = 2  # Runner

    frames.append(grid.copy())  # Save this frame
    steps += 1

# Setup the plot for animation
fig, ax = plt.subplots()
img = ax.imshow(frames[0], cmap="hot", interpolation="nearest")
plt.title("Tag Game - Chaser (1) vs Runner (2)")
plt.axis("off")

# Update function for the animation
def update(frame):
    img.set_data(frame)
    return [img]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, blit=True)

# Save as GIF
gif_path = r"D:\VS Stuff\AI\Tag\watch_output.gif"
ani.save(gif_path, writer='pillow', fps=2)

print(f"Episode ended. GIF saved to: {gif_path}")
