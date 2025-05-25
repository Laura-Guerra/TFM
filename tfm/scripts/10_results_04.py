import json
import pandas as pd
from pathlib import Path
from tfm.src.config.settings import PATH_DATA_RESULTS

# Ruta base dels resultats d'avaluació
base_path = PATH_DATA_RESULTS / "evaluations"
data = []

# Carrega resultats JSON
for agent in ["dqn", "ppo", "sac"]:
    for variant in ["with", "without"]:
        for iteration in range(31, 41):
            json_path = base_path / agent / variant / f"iteration_{iteration}" / "test_episode.json"
            if not json_path.exists():
                continue

            try:
                with open(json_path, "r") as f:
                    metrics = json.load(f)

                # if metrics["Max Drawdown"]== 0:
                #     print(f"⚠️ No hi ha retorn acumulat per {json_path}")
                #     continue

                metrics.update({
                    "agent": agent.upper(),
                    "with_news": variant == "with",
                    "iteration": iteration,
                    "Cumulative Return (%)": metrics.get("Cumulative Return", 0) * 100
                })
                data.append(metrics)

            except Exception as e:
                print(f"❌ Error amb {json_path}: {e}")

# Convertim a DataFrame
df = pd.DataFrame(data)

# Mètriques a agregar
metrics_to_agg = ["Reward Sum", "Sharpe Ratio", "Max Drawdown", "Cumulative Return (%)"]

# Agregat: mitjana, desviació i mediana
summary = (
    df.groupby(["agent", "with_news"])[metrics_to_agg]
    .agg(["mean", "std", "median"])
    .reset_index()
)

# Aplanem els noms de les columnes
summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

# Mostrem la taula
print("\n📊 Taula agregada amb mitjanes, desviacions i mediana:\n")
print(summary.to_string(index=False))

# %% Desa a CSV
summary.to_csv("taula_resultats_completa_metrics_finals.csv", index=False)
