# %%
import time
from datetime import datetime

from tfm.src.nlp.extraction import NewsExtractor


# %%
def generate_monthly_intervals(
        start_year: int, start_month: int
) -> list[tuple[int, int]]:
    today = datetime.today()
    current_year = today.year
    current_month = today.month

    intervals = []
    year, month = start_year, start_month

    while (year, month) <= (current_year, current_month):
        intervals.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return intervals


intervals = generate_monthly_intervals(2014, 1)
extractor = NewsExtractor()
for year, month in intervals:
    print(f"Processing {month}-{year}")
    try:
        extractor.extract_nyt_news(month, year)
    except Exception:
        print(f"Error while extracting news of {month}-{year}")
    time.sleep(12)
print("Finished")