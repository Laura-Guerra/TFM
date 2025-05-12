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
PATH_DATA_LOGS = ROOT_DIR / "tfm" / "data" / "logs"
PATH_DATA_RESULTS = ROOT_DIR / "tfm" / "data" / "results"
PATH_DATA_MODELS = ROOT_DIR / "tfm" / "data" / "models"



# Model settings
NEWS_SECTIONS = [
    "Business Day", "World", "U.S.", "Technology", "Your Money", "Briefing", "The Upshot", "Science"
]


NEWS_SUBSECTIONS = [
    "Economy",
    "DealBook",
    "International Business",
    "Stocks and Bonds",
    "Mutual Funds and ETFs",
    "Asset Allocation",
    "Energy & Environment",
    "Politics",
    "Elections",
    "Europe",
    "Americas",
    "Asia Pacific",
    "Middle East"
]