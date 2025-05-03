import numpy as np
import optuna
from pathlib import Path
from stable_baselines3 import DQN

from tfm.src.rl.agents.base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict = None):
        super().__init__(env, eval_env, model_dir, log_dir, params or {})
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

    def load(self, path: str):
        self.model = DQN.load(path, env=self.env)

    def optimize_hyperparameters(self, n_trials=30, n_eval_episodes=5):
        """
        Use Optuna to search for best hyperparameters.
        """
        def objective(trial):
            trial_params = {
                # log‑uniform en l’escala [1e‑5, 1e‑3]
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": 100_000,
                "learning_starts": trial.suggest_int("learning_starts", 100, 2_000, step=100),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),

                # uniforme simple
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),

                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
                "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),

                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
            }

            model = DQN(
                policy="MlpPolicy",
                env=self.env,
                tensorboard_log=str(self.log_dir),
                **trial_params
            )

            model.learn(total_timesteps=20_000)  # Quick fit
            mean_reward = self._evaluate_model(model, n_eval_episodes)

            return -mean_reward  # Optuna always minimizes

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.params = study.best_params
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

        print(f"✅ Optuna best params: {self.params}")
        return self.params

    def _evaluate_model(self, model, n_episodes: int = 5) -> float:
        """
        Returns average reward over `n_episodes`.
        If no episode could be completed, returns -inf so Optuna penalises it.
        """
        rewards = []

        for _ in range(n_episodes):
            try:
                obs, _ = self.eval_env.reset()
            except Exception as e:
                # Si reset falla, saltem episodi
                print(f"[WARN] reset() va fallar durant avaluació: {e}")
                continue

            done = False
            total_reward = 0.0

            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                except Exception as e:
                    # Qualsevol error en l'episodi → descartar‑lo
                    print(f"[WARN] episodi descartat per error: {e}")
                    total_reward = None
                    break

            if total_reward is not None:
                rewards.append(total_reward)

        # ── Retorn segur ─────────────────────────────────────────────
        if len(rewards) == 0:
            return float("-inf")  # Optuna ho penalitzarà
        return float(np.mean(rewards))
