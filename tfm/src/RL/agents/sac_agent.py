import os

import numpy as np
import optuna
from pathlib import Path
from stable_baselines3 import SAC
from tfm.src.RL.agents.base_agent import BaseAgent


class SACAgent(BaseAgent):
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict):
        super().__init__(env, eval_env, model_dir, log_dir, params)
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

    def load(self, path: str):
        self.model = SAC.load(path, env=self.env)

    def optimize_hyperparameters(self, n_trials: int = 30, n_timesteps: int = 10000):
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                "buffer_size": trial.suggest_int("buffer_size", 10_000, 100_000, step=10_000),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
                "tau": trial.suggest_uniform("tau", 0.005, 0.02),
                "train_freq": trial.suggest_int("train_freq", 1, 10),
            }

            model = SAC(
                "MlpPolicy",
                self.env,
                verbose=0,
                tensorboard_log=str(self.log_dir / "optuna_trials"),
                **params
            )
            model.learn(total_timesteps=n_timesteps)

            rewards = []
            for _ in range(5):
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    total_reward += reward
                rewards.append(total_reward)

            avg_reward = np.mean(rewards)
            return avg_reward

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_trial.params
        print("\nBest hyperparameters:", best_params)

        # Update model
        self.model = SAC(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            **best_params
        )
