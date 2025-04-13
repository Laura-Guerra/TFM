# %%
import os
from pathlib import Path

from dotenv import load_dotenv

# Carregar el .env
load_dotenv()

# Definim les variables que volem exposar
NEWS_API_KEY = os.getenv("NYT_API_KEY")
NEWS_URL = os.getenv("NYT_URL")


ROOT_DIR = Path(__file__).resolve().parents[3]

PATH_DATA_RAW = ROOT_DIR / "tfm" / "data" / "raw"
PATH_DATA_PROCESSED = ROOT_DIR / "tfm" / "data" / "processed"