from loguru import logger
import numpy as np
import optuna
from optuna.trial import FrozenTrial
from pathlib import Path
from stable_baselines3 import SAC

from tfm.src.rl.agents.base_agent import BaseAgent


class SACAgent(BaseAgent):
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict = None):
        super().__init__(env, eval_env, model_dir, log_dir, params or {})

        logger.info("🔧 Instanciant model inicial amb paràmetres per defecte")
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

    def optimize_hyperparameters(self, n_trials: int = 30, n_eval_episodes: int = 5):
        """
        Cerqueu hiperparàmetres òptims amb Optuna.
        """

        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": 100_000,
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
                "tau": trial.suggest_float("tau", 0.005, 0.02),
                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
                "ent_coef": trial.suggest_categorical("ent_coef", ["auto", "auto_0.1", "auto_0.01"]),
            }

            logger.debug(f"🧪 Trial {trial.number}: {trial_params}")

            model = SAC(
                policy="MlpPolicy",
                env=self.env,
                tensorboard_log=str(self.log_dir),
                **trial_params
            )
            model.learn(total_timesteps=20_000, progress_bar=False)

            mean_reward = self._evaluate_optuna_model(model, n_eval_episodes)
            logger.debug(f"🎯 Trial {trial.number} — MeanReward={mean_reward:.2f}")

            return -mean_reward

        def log_callback(study: optuna.Study, trial: FrozenTrial):
            if study.best_trial == trial:
                logger.info(f"🏅  Nou millor trial {trial.number}  →  reward = {-trial.value:.2f}")

        logger.info(f"🔍 Cerca d'hiperparàmetres ({n_trials} trials)…")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[log_callback])

        self.params = study.best_params
        logger.success(f"✅ Millors hiperparàmetres trobats: {self.params}")

        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )
        return self.params

    def load(self, checkpoint_path: str, total_timesteps: int = 100_000):
        logger.info(f"🔄 Carregant model des de {checkpoint_path}…")
        self.model = SAC.load(checkpoint_path, env=self.env, tensorboard_log=str(self.log_dir))

    def _evaluate_optuna_model(self, model, n_episodes: int = 5) -> float:
        rewards = []

        for ep in range(1, n_episodes + 1):
            try:
                obs, _ = self.eval_env.reset()
            except Exception as e:
                logger.warning(f"[EP {ep}] reset() ha fallat: {e}")
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
                    logger.warning(f"[EP {ep}] episodi interromput: {e}")
                    total_reward = None
                    break

            if total_reward is not None:
                logger.debug(f"[EP {ep}] recompensa = {total_reward:.2f}")
                rewards.append(total_reward)

        if len(rewards) == 0:
            logger.error("🚫 Cap episodi completat — retorn -inf")
            return float("-inf")

        mean_r = float(np.mean(rewards))
        logger.debug(f"→ Recompensa mitjana ({len(rewards)} eps) = {mean_r:.2f}")
        return mean_r
