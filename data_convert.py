# Convert hourly data into daily average data
import numpy as np
from pathlib import Path
import pandas as pd

DATA_FOLDER = Path("./download")
FILE_LIST = list(DATA_FOLDER.glob("*.csv"))

daily_avg = pd.DataFrame()
daily_avg_list = []

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

    df['Date'] = file_date  # Date
    df['TempMax'] = df['Temperature'].max() # Maximum Temperature
    df['TempMin'] = df['Temperature'].min() # Minimum Temperature
    df['Dew'] = df['Temperature'] - (100 - df['RH']) / 5 # Dew point
    df['PrecpType'] = df['Precp'].apply(lambda x: 'rain' if x > 0.2 else np.NAN)

    df = df[['Date','TempMax','TempMin',"Temperature","Dew","RH","Precp", "PrecpType","WS","WD","StnPres"]]

    daily_avg_list.append(df)

daily_avg = pd.concat(daily_avg_list, ignore_index=True)
daily_avg['Month'] = daily_avg['Date'].dt.month
daily_avg = daily_avg.groupby('Date').agg({
    'Month': 'first',
    'TempMax': 'mean',
    'TempMin': 'mean',
    'Temperature': 'mean',
    'Dew': 'mean',
    'RH': 'mean',
    'Precp': 'mean',
    'PrecpType': 'first',  # Don't calculate mean value of column `PrecpType`
    'WS': 'mean',
    'WD': 'mean',
    'StnPres': 'mean'
}).reset_index()

print(daily_avg)

OUTPUT_FOLDER = Path("./data")
OUTPUT_NAME = Path("daily_average.csv")
OUTPUT_PATH = OUTPUT_FOLDER / OUTPUT_NAME

# Check if output folder exist
if not OUTPUT_FOLDER.is_dir():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

daily_avg.to_csv(OUTPUT_PATH, index=True)

print(f"Average data saved to {OUTPUT_PATH}")