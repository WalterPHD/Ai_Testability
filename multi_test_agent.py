from stable_baselines3 import PPO
from tag_env import TagEnv
import os
import multiprocessing
import numpy as np
import time

# Define the save path
save_dir = r"D:\VS Stuff\AI\Tag\train"

# Function to evaluate one test and return the result
def evaluate_model(model_path, env, max_steps=100, steps_per_episode=100):
    # Load the trained model
    model = PPO.load(model_path)

    best_result = {"survival_time": 0, "steps": 0, "episode": 0}
    
    for episode in range(steps_per_episode):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            steps += 1

            # Check if the runner survived longer
            if reward == 1.0:
                best_result["survival_time"] = steps
                best_result["steps"] = steps
                best_result["episode"] = episode
                break

        time.sleep(0.5)  # Simulate a brief delay for visualization if needed

    return best_result


# Create the pool of processes
def run_multiple_tests(model_path, num_tests=4, steps_per_episode=10):
    env = TagEnv()
    pool = multiprocessing.Pool(processes=num_tests)

    # Run multiple tests at once
    results = [pool.apply_async(evaluate_model, (model_path, env, 100, steps_per_episode)) for _ in range(num_tests)]
    pool.close()
    pool.join()

    # Get the best result from all tests
    best_outcome = None
    for res in results:
        result = res.get()  # Wait for the process to complete and get the result
        if best_outcome is None or result["survival_time"] > best_outcome["survival_time"]:
            best_outcome = result
    
    return best_outcome

if __name__ == "__main__":
    # Train the model first
    print("Training the model...")
    env = TagEnv()
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=100_000)

    # Save the model
    print(f"Saving model to {save_dir}...")
    model.save(os.path.join(save_dir, "tag_model"))

    # Run multiple tests
    print("Running multiple tests to evaluate the best outcome...")
    best_result = run_multiple_tests(os.path.join(save_dir, "tag_model"), num_tests=4)

    print(f"Best test outcome: {best_result}")
