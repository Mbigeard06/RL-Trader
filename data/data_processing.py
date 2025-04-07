import pandas as pd
import glob
import os


data_dir = "./data/archive"
all_files = glob.glob(os.path.join(data_dir, "*.csv"))
print(all_files)
df_list = []

#Navigate through all the files
for file in all_files:
    ticker = os.path.splitext(os.path.basename(file))[0]
    temp_df = pd.read_csv(file, parse_dates=["date"])
    df_list.append(temp_df)


df = pd.concat(df_list)
df.sort_values(by=["ticker", "date"], inplace=True)

# Save DataFrame to CSV
save_path = "./data/sorted_nyse_data.csv"
df.to_csv(save_path, index=False)

