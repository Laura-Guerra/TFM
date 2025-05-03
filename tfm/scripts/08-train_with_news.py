# train_dqn_with_optuna.py
# ──────────────────────────────────────────────────────────────
import pandas as pd
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.dqn_agent import DQNAgent              # <- la classe que has creat
from tfm.src.config.settings import (
    PATH_DATA_PROCESSED,
    PATH_DATA_LOGS,
    PATH_DATA_MODELS,
)

# 1) Carregar les dades ───────────────────────────────────────
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv")
df = df[df["ticker"] == "SPY"].reset_index(drop=True)

# 2) Crear entorn d’entrenament i d’avaluació ────────────────
train_env = StockEnvironment(df, initial_balance=10_000, continuous_actions=False)
eval_env  = StockEnvironment(df.copy(), initial_balance=10_000, continuous_actions=False)

# (opcional) limitem la durada màxima d’un episodi per accelerar
MAX_STEPS = 1_000
train_env = TimeLimit(train_env, MAX_STEPS)
eval_env  = TimeLimit(eval_env,  MAX_STEPS)

# 3) Carpetes on desarem logs i models ────────────────────────
run_name  = "dqn_spy_optuna"
log_dir   = PATH_DATA_LOGS   / run_name
model_dir = PATH_DATA_MODELS / run_name
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

# 4) Crear l’agent ────────────────────────────────────────────
agent = DQNAgent(
    env=train_env,
    eval_env=eval_env,
    model_dir=model_dir,
    log_dir=log_dir,
)


# 5) Buscar hiperparàmetres amb Optuna ────────────────────────
agent.optimize_hyperparameters(n_trials=30, n_eval_episodes=5)

# 6) Entrenament final ────────────────────────────────────────
agent.train(total_timesteps=500_000)
agent.save("dqn_final")

# 7) Guardar el model ────────────────────────────────────────
print(f"✅ Model final desat a {model_dir/'dqn_final.zip'}")
print(f"📊 Pots obrir TensorBoard amb: tensorboard --logdir {log_dir}")
