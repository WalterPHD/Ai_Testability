from stable_baselines3 import PPO
from tag_env import TagEnv
import os
import multiprocessing
import time

save_dir = r"D:\VS Stuff\AI\Tag\train"
model_path = os.path.join(save_dir, "tag_model.zip")
os.makedirs(save_dir, exist_ok=True)

# Train model
print("Initializing environment and model...")
env = TagEnv()
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

print("Training model...")
model.learn(total_timesteps=100_000)

print(f"Saving model to {model_path}...")
model.save(model_path)

# Evaluation function
def evaluate_model(model_path, max_steps=100, steps_per_episode=100):
    model = PPO.load(model_path)
    env = TagEnv()
    best_result = {"survival_time": 0, "steps": 0, "episode": 0}

    for episode in range(steps_per_episode):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            steps += 1

        if steps > best_result["survival_time"]:
            best_result.update({
                "survival_time": steps,
                "steps": steps,
                "episode": episode
            })

    return best_result

# Run multiple tests in parallel
def run_multiple_tests(model_path, num_tests=4, steps_per_episode=10):
    pool = multiprocessing.Pool(processes=num_tests)
    results = [pool.apply_async(evaluate_model, (model_path, 100, steps_per_episode)) for _ in range(num_tests)]
    pool.close()
    pool.join()

    best_outcome = None
    for res in results:
        result = res.get()
        if not best_outcome or result["survival_time"] > best_outcome["survival_time"]:
            best_outcome = result

    return best_outcome

print("Evaluating model with multiple test runs...")
best_result = run_multiple_tests(model_path)
print(f"Best test outcome: {best_result}")
