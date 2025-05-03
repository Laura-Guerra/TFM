import pandas as pd
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv

from tfm.src.rl.stock_env import StockEnvironment
from tfm.src.rl.agents.dqn_agent import DQNAgent
from tfm.src.config.settings import PATH_DATA_PROCESSED

# -----------------------------
# Cargar datos y filtrar ticker
# -----------------------------
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv")
df = df[df["ticker"] == "SPY"]  # Solo un activo

# ------------------------------------
# Crear entornos (solo uno vectorizado)
# ------------------------------------
env = DummyVecEnv([lambda: StockEnvironment(
    df=df,
    initial_balance=100000.0,
    continuous_actions=False,
)])

eval_env = DummyVecEnv([lambda: StockEnvironment(
    df=df,
    initial_balance=100000.0,
    continuous_actions=False,
)])

# ------------------------
# Definir directorios
# ------------------------
model_dir = Path("models/dqn_test")
log_dir = Path("logs/dqn_test")

# ------------------------
# Hiperparámetros de prueba
# ------------------------
params = {
    "learning_rate": 1e-4,
    "buffer_size": 10000,
    "learning_starts": 500,
    "batch_size": 32,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 250,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.05
}

# ------------------------
# Crear e iniciar el agente
# ------------------------
print("env class:", type(env))
print("eval_env class:", type(eval_env))
agent = DQNAgent(env, eval_env, model_dir, log_dir, params=params)

# ------------------------
# Entrenamiento corto
# ------------------------
print("🚀 Starting short training...")
agent.model.learn(total_timesteps=1000)
print("✅ Training completed.")

# ------------------------
# Evaluación
# ------------------------
print("🔍 Evaluating agent...")
agent.evaluate(n_episodes=1)
