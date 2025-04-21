import pandas as pd


def compute_effective_market_date(pub_date_str: str, market: str = "us") -> pd.Timestamp:
    """
    Given a UTC pub_date string, returns the market-effective date for US or EU,
    based on local market open time and a 1-hour cutoff before it.

    Args:
        pub_date_str: ISO-format string with timezone info (e.g., "2024-06-01T01:55:03+0000")
        market: Either "us" (New York) or "eu" (Madrid)

    Returns:
        pd.Timestamp.date object representing the market day the news affects.
    """
    market_settings = {
        "us": {
            "timezone": "America/New_York",
            "open_hour": 9,
            "open_minute": 30
        },
        "eu": {
            "timezone": "Europe/Madrid",
            "open_hour": 9,
            "open_minute": 0
        }
    }

    if market not in market_settings:
        raise ValueError("Market must be 'us' or 'eu'.")

    settings = market_settings[market]

    # Parse timestamp
    pub_dt_utc = pd.to_datetime(pub_date_str).tz_convert("UTC")
    pub_dt_local = pub_dt_utc.tz_convert(settings["timezone"])
    pub_date_only = pub_dt_local.date()

    # Build datetime for market open (same day)
    market_open = pd.Timestamp(
        year=pub_dt_local.year,
        month=pub_dt_local.month,
        day=pub_dt_local.day,
        hour=settings["open_hour"],
        minute=settings["open_minute"],
        tz=settings["timezone"]
    )

    # 1 hour before open
    cutoff_time = market_open - pd.Timedelta(hours=1)

    if pub_dt_local >= cutoff_time:
        return (pub_dt_local + pd.Timedelta(days=1)).date()
    else:
        return pub_date_only