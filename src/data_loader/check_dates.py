import pandas as pd
import pandas_market_calendars as mcal


def validate_trading_data(df, calendar_name="NYSE", date_col="date", ticker_col="ticker"):
    df = df.copy()
    df = df.sort_values(by=[ticker_col, date_col]).reset_index(drop=True)

    cal = mcal.get_calendar(calendar_name)

    # Dictionnaire pour stocker les jours manquants par ticker
    missing_dates_by_ticker = {}

    for ticker, group in df.groupby(ticker_col):
        start = group[date_col].min()
        end = group[date_col].max()

        valid_days = cal.valid_days(start_date=start, end_date=end).tz_localize(None)
        actual_dates = pd.Series(group[date_col].unique())
        missing = sorted(set(valid_days) - set(actual_dates))
        if missing:
            missing_dates_by_ticker[ticker] = missing

    return missing_dates_by_ticker


df = pd.read_csv("../../data/blueChips/IBM.csv", parse_dates=["datetime"])
df["ticker"] = "IBM"
missing_dates = validate_trading_data(df, date_col = "datetime")

if not missing_dates:
    print("✅ Aucune date manquante détectée !")
for ticker, dates in missing_dates.items():
    print(f"\n{ticker} is missing {len(dates)} trading days:")