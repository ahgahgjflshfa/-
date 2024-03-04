# Convert hourly data into daily average data

from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("./download")
FILE_LIST = list(DATA_FOLDER.glob("*.csv"))

daily_avg = pd.DataFrame()

empty = []
for file in FILE_LIST:
    # Extract date from file name
    date_str = "".join(file.stem.split("-")[1:])
    file_date = pd.to_datetime(date_str, format="%Y%m%d")

    # Read csv file
    try:
        df = pd.read_csv(file, skiprows=1, na_values=['--', '\\', '/', '&', 'X'])   # fucking stupid
    except pd.errors.EmptyDataError as e:
        empty.append(file)
        continue

    df['Date'] = file_date

    df = df[['Date',"StnPres","Temperature","RH","WS","WD","Precp"]]

    daily_avg = pd.concat([daily_avg, df])


daily_avg = daily_avg.groupby('Date').mean()

OUTPUT_FOLDER = Path("./data")
OUTPUT_NAME = Path("daily_average.csv")
OUTPUT_PATH = OUTPUT_FOLDER / OUTPUT_NAME

# Check if output folder exist
if not OUTPUT_FOLDER.is_dir():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

daily_avg.to_csv(OUTPUT_PATH, index=True)

print(f"Average data saved to {OUTPUT_PATH}")