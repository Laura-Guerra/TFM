import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tfm.src.config.settings import PATH_DATA_RESULTS
from tfm.src.viz.plot_theme import set_custom_theme
set_custom_theme()

# --- Configuració ---
AGENTS = ["dqn", "ppo", "sac"]
VARIANT = "without"
BASE_PATH = PATH_DATA_RESULTS / "evaluations"

metrics = []

for agent in AGENTS:
    agent_dir = BASE_PATH / agent / VARIANT
    for iteration in sorted(agent_dir.glob("iteration_*")):
        metrics_path = iteration / "test_episode.json"
        if not metrics_path.exists():
            continue
        try:
            with open(metrics_path) as f:
                values = pd.json_normalize([json.load(f)])
                values["agent"] = agent.upper()
                values["iteration"] = iteration.name
                metrics.append(values)
        except Exception as e:
            print(f"⚠️ Error amb {metrics_path}: {e}")

# Concatena totes les files
df = pd.concat(metrics, ignore_index=True)

# --- Gràfiques: scatter amb esferes i línia horitzontal de mitjana ---
agents = df["agent"].unique()
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Sharpe Ratio filtrant per drawdown > 0
for i, agent in enumerate(agents):
    sub_df = df[(df["agent"] == agent) & (df["Max Drawdown"] > 0)]
    values = sub_df["Sharpe Ratio"]
    mean = values.mean()
    jitter = np.random.normal(0, 0.05, size=len(values))
    axs[0].scatter(np.full_like(values, i, dtype=float) + jitter, values, s=80, alpha=0.5)
    axs[0].hlines(mean, i - 0.3, i + 0.3, linewidth=2, label="Mitjana" if i == 0 else None)

axs[0].set_xticks(range(len(agents)))
axs[0].set_xticklabels(agents)
axs[0].set_ylabel("Sharpe Ratio")
axs[0].legend()

# Max Drawdown (filtrem valors == 0 com abans)
for i, agent in enumerate(agents):
    values = df[(df["agent"] == agent)]["Max Drawdown"]
    values = values[values > 0]
    mean = values.mean()
    jitter = np.random.normal(0, 0.05, size=len(values))
    axs[1].scatter(np.full_like(values, i, dtype=float) + jitter, values, s=80, alpha=0.5)
    axs[1].hlines(mean, i - 0.3, i + 0.3, linewidth=2)

axs[1].set_xticks(range(len(agents)))
axs[1].set_xticklabels(agents)
axs[1].set_ylabel("Drawdown màxim")

# Desa i mostra
plt.savefig(PATH_DATA_RESULTS / "figures" / f"scatter_meanbar_sharpe_drawdown_{VARIANT}.png")
plt.show()
