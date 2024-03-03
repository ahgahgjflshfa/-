# Convert hourly data into daily average data

from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("./download")
FILE_LIST = list(DATA_FOLDER.glob("*.csv"))

daily_avg = pd.DataFrame()

empty = []
for file in FILE_LIST:
    # Read csv file
    try:
        df = pd.read_csv(file, skiprows=1, na_values=['--', '\\', '/', '&', 'X'])   # fucking stupid
    except pd.errors.EmptyDataError:
        empty.append(file)
        continue

    avg_values = df[["StnPres","Temperature","RH","WS","WD","Precp"]].mean()

    daily_avg = pd.concat([daily_avg, avg_values], axis=1)

daily_avg = daily_avg.transpose()

OUTPUT_FOLDER = Path("./data")
OUTPUT_NAME = Path("daily_average.csv")
OUTPUT_PATH = OUTPUT_FOLDER / OUTPUT_NAME

# Check if output folder exist
if not OUTPUT_FOLDER.is_dir():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

daily_avg.to_csv(OUTPUT_PATH, index=False, lineterminator='')

print(f"Average data saved to {OUTPUT_PATH}")