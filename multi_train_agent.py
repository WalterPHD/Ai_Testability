from stable_baselines3 import PPO
from tag_env import TagEnv
import os
import multiprocessing
import time
import shutil
import numpy as np

# Define the save path
save_dir = r"D:\VS Stuff\AI\Tag\train"

# Function to evaluate a trained model
def evaluate_model(model, max_steps=100):
    env = TagEnv()
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward

# Function to train and evaluate one model
def train_model(model_id, total_timesteps=100_000, max_steps=100):
    print(f"Training model {model_id}...")
    env = TagEnv()
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    model.learn(total_timesteps=total_timesteps)

    model_filename = f"tag_model_{model_id}.zip"
    model_path = os.path.join(save_dir, model_filename)
    print(f"Saving model {model_id} to {model_path}...")
    model.save(model_path)

    reward = evaluate_model(model, max_steps)
    return model_path, reward

# Train multiple models in parallel
def run_multiple_trainings(num_trainings=4, total_timesteps=100_000, max_steps=100):
    pool = multiprocessing.Pool(processes=num_trainings)
    results = [pool.apply_async(train_model, (i, total_timesteps, max_steps)) for i in range(num_trainings)]
    pool.close()
    pool.join()

    best_model_path = None
    best_reward = -float('inf')

    for res in results:
        model_path, reward = res.get()
        print(f"Model trained at {model_path} with reward: {reward}")
        if reward > best_reward:
            best_reward = reward
            best_model_path = model_path

    return best_model_path, best_reward

if __name__ == "__main__":
    print("Running multiple training sessions...")
    os.makedirs(save_dir, exist_ok=True)

    best_model_path, best_reward = run_multiple_trainings()

    print(f"Best model path: {best_model_path} with reward: {best_reward}")
    best_model_save_path = os.path.join(save_dir, "best_tag_model.zip")
    shutil.copy(best_model_path, best_model_save_path)
    print(f"Best model saved at {best_model_save_path}")
