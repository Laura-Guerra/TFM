from loguru import logger
import numpy as np
import optuna
from optuna.trial import FrozenTrial
from pathlib import Path
from stable_baselines3 import DQN

from tfm.src.rl.agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict = None):
        super().__init__(env, eval_env, model_dir, log_dir, params or {})

        logger.info("🔧 Instanciant model inicial amb paràmetres per defecte")
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

    # ────────────────────────────────────────────────────────────────
    # Optuna Hyper‑parameter Search
    # ────────────────────────────────────────────────────────────────
    def optimize_hyperparameters(self, n_trials: int = 30, n_eval_episodes: int = 5):
        """
        Cerqueu hiperparàmetres òptims amb Optuna.
        """

        # ── Definim l'objectiu ─────────────────────────────────────
        def objective(trial: optuna.Trial) -> float:
            trial_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": 100_000,
                "learning_starts": trial.suggest_int("learning_starts", 100, 2_000, step=100),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
                "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
                "target_update_interval": trial.suggest_categorical("target_update_interval", [500, 1_000, 5_000]),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
            }

            logger.debug(f"🧪 Trial {trial.number}: {trial_params}")

            # Entrenament ràpid
            model = DQN(
                policy="MlpPolicy",
                env=self.env,
                tensorboard_log=str(self.log_dir),
                **trial_params
            )
            model.learn(total_timesteps=20_000, progress_bar=False)

            # Avaluem
            mean_reward = self._evaluate_model(model, n_eval_episodes)
            logger.debug(f"🎯 Trial {trial.number} — MeanReward={mean_reward:.2f}")

            # Optuna minimitza ⇒ retornem negatiu
            return -mean_reward

        # ── Callback per mostrar millors resultats al vol ───────────
        def log_callback(study: optuna.Study, trial: FrozenTrial):
            if study.best_trial == trial:
                logger.info(f"🏅  Nou millor trial {trial.number}  →  "
                            f"reward = {-trial.value:.2f}")

        logger.info(f"🔍 Començo la cerca d'hiperparàmetres ({n_trials} trials)…")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[log_callback])

        # ── Guardem i reconstruïm el model amb els millors paràmetres ──
        self.params = study.best_params
        logger.success(f"✅ Millors hiperparàmetres trobats: {self.params}")

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )
        return self.params

    # ────────────────────────────────────────────────────────────────
    # Avaluació d'un model
    # ────────────────────────────────────────────────────────────────
    def _evaluate_model(self, model, n_episodes: int = 5) -> float:
        """
        Retorna la recompensa mitjana en `n_episodes`.
        Si cap episodi es completa, retorna -inf perquè Optuna el penalitzi.
        """
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
