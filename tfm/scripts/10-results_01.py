# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tfm.src.config.settings import PATH_DATA_PROCESSED
from tfm.src.viz.plot_theme import set_matplotlib_theme

# %%
set_matplotlib_theme()

# %%
# Carrega dades i calcula la mitjana mensual
df = pd.read_csv(PATH_DATA_PROCESSED / "state_features.csv", parse_dates=["date"])
topic_cols = [col for col in df.columns if col.startswith("lda_")]
df_lda = df[["date"] + topic_cols].copy().set_index("date")

df_lda_monthly = df_lda.resample("M").mean()
df_lda_monthly = df_lda_monthly[df_lda_monthly.index >= "2024-01-01"]

# Diccionari de noms llegibles
topic_labels = {col: f"Tema {i+1}" for i, col in enumerate(topic_cols)}
first_half = topic_cols[:5]
second_half = topic_cols[5:]

# %%
# Multiplot horitzontal
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Subgràfica esquerra (temes 1–5)
for col in first_half:
    axes[0].plot(df_lda_monthly.index, df_lda_monthly[col], label=topic_labels[col])
axes[0].set_xlabel("Data")
axes[0].set_ylabel("Proporció mitjana de tema")
axes[0].legend(loc="upper left", fontsize=9)

# Subgràfica dreta (temes 6–10)
for col in second_half:
    axes[1].plot(df_lda_monthly.index, df_lda_monthly[col], label=topic_labels[col])
axes[1].set_xlabel("Data")
axes[1].legend(loc="upper left", fontsize=9)

# Títol global
fig.suptitle("Evolució mensual de la distribució temàtica (LDA)", fontsize=16)

# Ajustos
plt.tight_layout(rect=[0, 0, 1, 0.95])  # deixa espai per al títol superior
plt.savefig(PATH_DATA_PROCESSED / "fig_lda_lineplot_mensual_subplots.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Subset i etiqueta numèrica de trimestre
X = df_lda[topic_cols].copy()
X = X[X.index >= "2024-01-01"].drop_duplicates()
quarter_str = X.index.to_period("Q").astype(str)
quarter_numeric = pd.factorize(quarter_str)[0]  # converteix a 0,1,2,...

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Gràfic amb colormap
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=quarter_numeric, cmap="viridis", s=50)
plt.colorbar(sc, ticks=range(len(set(quarter_numeric))), label="Trimestre a partir de 2024")
plt.title("t-SNE de distribucions temàtiques")
plt.xlabel("Dimensió 1")
plt.ylabel("Dimensió 2")
plt.tight_layout()
plt.savefig(PATH_DATA_PROCESSED / "fig_lda_tsne_trimestres_colormap.png", dpi=300)
plt.show()

# %%
# t-SNE sobre vectors Word2Vec (w2v_), colorejat per trimestre amb reducció prèvia amb PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Filtra columnes w2v_
w2v_cols = [col for col in df.columns if col.startswith("w2v_")]
df_w2v = df[["date"] + w2v_cols].copy().set_index("date")

# Filtra període de test
df_w2v_filtered = df_w2v[df_w2v.index >= "2024-01-01"]

# Etiquetes de trimestre (com a valor numèric)
quarter_str = df_w2v_filtered.index.to_period("Q").astype(str)
quarter_numeric = pd.factorize(quarter_str)[0]

# Reducció prèvia amb PCA
X_w2v = df_w2v_filtered.values
X_pca = PCA(n_components=30, random_state=42).fit_transform(X_w2v)

# t-SNE optimitzat
tsne = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate=150,
    n_iter=800,
    init="random",
    random_state=42
)
X_embedded_w2v = tsne.fit_transform(X_pca)

# Gràfic
plt.figure(figsize=(10, 6))
sc = plt.scatter(X_embedded_w2v[:, 0], X_embedded_w2v[:, 1], c=quarter_numeric, cmap="viridis", s=50, alpha=0.85)
plt.colorbar(sc, ticks=range(len(set(quarter_numeric))), label="Trimestre a partir de 2024")
plt.title("t-SNE dels vectors Word2Vec agregats diàriament (2024–2025)", fontsize=14)
plt.xlabel("Dimensió 1")
plt.ylabel("Dimensió 2")
plt.tight_layout()
plt.savefig(PATH_DATA_PROCESSED / "fig_w2v_tsne_trimestres_colormap_pca.png", dpi=300)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepara les dades
df_sent = df[["date", "sentiment_mean_us", "sentiment"]].copy()
df_sent["quarter"] = df_sent["date"].dt.to_period("Q").astype(str)
df_sent = df_sent[df_sent["quarter"] >= "2024Q1"].drop_duplicates()

# Inicialitza la figura
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # format horitzontal

# Gràfic 1: sentiment mitjà (eixos invertits)
sns.violinplot(
    data=df_sent,
    y="quarter",
    x="sentiment_mean_us",
    ax=axes[0],
    inner="quartile"
)
axes[0].set_title("Sentiment mitjà per trimestre")
axes[0].set_ylabel("Trimestre")
axes[0].set_xlabel("Valor de sentiment")

# Gràfic 2: sentiment global (eixos invertits)
sns.violinplot(
    data=df_sent,
    y="quarter",
    x="sentiment",
    ax=axes[1],
    palette="pastel",
    inner="quartile"

)
axes[1].set_title("Sentiment global per trimestre")
axes[1].set_ylabel("")  # ja indicat al primer
axes[1].set_xlabel("Valor de sentiment")
axes[1].set_yticklabels([])  # elimina les etiquetes de l'eix y

# Títol global i ajust
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(PATH_DATA_PROCESSED / "fig_sentiment_violin_horizontal.png", dpi=300)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepara les dades
df_sent = df[["date", "sentiment_mean_us", "sentiment"]].copy()
df_sent["quarter"] = df_sent["date"].dt.to_period("Q").astype(str)
df_sent = df_sent[df_sent["quarter"] >= "2024Q1"].drop_duplicates()

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

# Sentiment mitjà
sns.boxplot(data=df_sent, x="quarter", y="sentiment_mean_us", ax=axes[0])
sns.swarmplot(data=df_sent, x="quarter", y="sentiment_mean_us", ax=axes[0], size=3,color='#73BAFF', alpha=0.6)
axes[0].set_title("Sentiment mitjà")
axes[0].set_ylabel("Valor de sentiment mitjà")
axes[0].set_xlabel("Trimestre")

# Sentiment global
sns.boxplot(data=df_sent, x="quarter", y="sentiment", ax=axes[1])
sns.swarmplot(data=df_sent, x="quarter", y="sentiment", ax=axes[1], size=3,color='#73BAFF', alpha=0.6)
axes[1].set_title("Sentiment global")
axes[1].set_ylabel("Valor de sentiment global")
axes[1].set_xlabel("Trimestre")

plt.tight_layout()
plt.savefig(PATH_DATA_PROCESSED / "fig_sentiment_swarm_boxplot.png", dpi=300)
plt.show()
