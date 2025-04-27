
import os
import pandas as pd
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

class BaseAgent:
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict):
        self.env = env
        self.eval_env = eval_env
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.params = params

        self.model = None  # Defined in subclass
        self.model_name = self.__class__.__name__.replace("Agent", "").lower()

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train(self, total_timesteps: int):
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,  # Every 10k steps
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix=self.model_name
        )

        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=10_000,
            deterministic=True,
            render=False
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback]
        )

    def save(self, filename: str = None):
        filename = filename or f"{self.model_name}_final"
        self.model.save(self.model_dir / filename)

    def load(self, path: str):
        raise NotImplementedError("Subclasses must implement load method!")

    def evaluate(self, n_episodes: int = 5):
        results = []
        for episode in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                total_reward += reward
            results.append(total_reward)

        avg_reward = sum(results) / len(results)
        print(f"\nAverage Reward over {n_episodes} episodes: {avg_reward:.2f}")

        df = pd.DataFrame({"episode_reward": results})
        df.to_csv(self.log_dir / "evaluation.csv", index=False)

    def optimize_hyperparameters(self):
        raise NotImplementedError("Optuna optimization to be implemented!")
