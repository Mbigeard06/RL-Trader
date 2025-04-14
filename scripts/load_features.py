from pathlib import Path
import pandas as pd

from rl_trader.data_loader.process_features import compute_features

def load_features(input, output):
    input_dir = Path(input)
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)

    for file in input_dir.glob("*.csv"):
        ticker = file.stem  # ← stem is a property, not a method
        print(f"Processing {ticker}...")

        try:
            df = pd.read_csv(file, parse_dates=["datetime"])  # make sure datetime is parsed
            features = compute_features(df=df)
            features.to_parquet(output_dir / f"{ticker}_features.parquet", index=False)
        except Exception as e:
            print(f"❌ Failed on {ticker}: {e}")

load_features("data/blueChips", "data/features/blueChips" )