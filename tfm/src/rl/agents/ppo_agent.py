import optuna
from pathlib import Path
from stable_baselines3 import PPO
from tfm.src.rl.agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, env, eval_env, model_dir: Path, log_dir: Path, params: dict = None):
        super().__init__(env, eval_env, model_dir, log_dir, params or {})
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)

    def optimize_hyperparameters(self, n_trials=30, n_eval_episodes=5):
        """
        Use Optuna to search for best hyperparameters.
        """
        def objective(trial):
            trial_params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
                "gae_lambda": trial.suggest_uniform("gae_lambda", 0.8, 1.0),
                "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
            }

            model = PPO(
                policy="MlpPolicy",
                env=self.env,
                tensorboard_log=str(self.log_dir),
                **trial_params
            )

            model.learn(total_timesteps=20_000)
            mean_reward = self._evaluate_model(model, n_eval_episodes)

            return -mean_reward  # Optuna always minimizes

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.params = study.best_params
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=str(self.log_dir),
            **self.params
        )

        print(f"\u2705 Optuna best params: {self.params}")
        return self.params

    def _evaluate_model(self, model, n_episodes: int):
        """
        Internal helper to evaluate a model during Optuna search.
        """
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        return sum(rewards) / len(rewards)
