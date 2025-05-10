
import os
from datetime import datetime

from loguru import logger

import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from tfm.src.config.settings import PATH_DATA_RESULTS


class BaseAgent:
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict, with_news: bool = False):
        self.env = env
        self.eval_env = eval_env
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.params = params

        self.model = None  # Defined in subclass
        self.model_name = self.__class__.__name__.replace("Agent", "").lower()
        self.with_news = with_news

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


        if with_news:
            self.model_name += "_news"
            self.log_dir = self.log_dir / "news"
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.model_dir / f"{self.model_name}_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, total_timesteps: int):
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.checkpoint_dir),
            name_prefix=self.model_name
        )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback]
        )

    def save(self, filename: str = None):
        filename = filename or f"{self.model_name}_final"
        self.model.save(self.model_dir / filename)

    def load(self, path: str):
        raise NotImplementedError("Subclasses must implement load method!")

    def evaluate(self, n_episodes: int = 5):
        folder = f"{self.model_name}_without_news" if not self.with_news else f"{self.model_name}"
        results_path = PATH_DATA_RESULTS / folder / datetime.today().strftime("%Y-%m-%d")
        results_path.mkdir(parents=True, exist_ok=True)
        episode_rewards = []
        history_paths = []

        for episode in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)

            # Guardar history per aquest episodi
            episode_path = results_path / f"episode_{episode + 1}.csv"
            self.eval_env.env.save_history(episode_path)
            history_paths.append(episode_path)

        # Guardar summary
        df_summary = pd.DataFrame({
            "episode": list(range(1, n_episodes + 1)),
            "reward": episode_rewards
        })
        df_summary.to_csv(self.log_dir / "evaluation_summary.csv", index=False)

        # Localitzar millor i pitjor episodis
        best_idx = np.argmax(episode_rewards)
        worst_idx = np.argmin(episode_rewards)

        # Guardar còpia del millor i pitjor
        best_path = self.log_dir / "best_episode.csv"
        worst_path = self.log_dir / "worst_episode.csv"

        pd.read_csv(history_paths[best_idx]).to_csv(best_path, index=False)
        pd.read_csv(history_paths[worst_idx]).to_csv(worst_path, index=False)

        print(f"✅ Evaluation done. Avg Reward: {np.mean(episode_rewards):.2f}")

    def optimize_hyperparameters(self):
        raise NotImplementedError("Optuna optimization to be implemented!")
