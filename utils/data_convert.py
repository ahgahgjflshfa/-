import pandas as pd
import numpy as np
from pathlib import Path

def data_convert(DATA_FOLDER: Path=Path("../download")):
    FILE_LIST = list(DATA_FOLDER.glob("*.csv"))

    daily_avg_list = []

    empty = []
    for file in FILE_LIST:
        # Extract date from file name
        date_str = "".join(file.stem.split("-")[1:])
        file_date = pd.to_datetime(date_str, format="%Y%m%d")

        # Read csv file
        try:
            df = pd.read_csv(file, skiprows=1, na_values=['--', '\\', '/', '&', 'X', ' '])   # fucking stupid
        except pd.errors.EmptyDataError as _:
            empty.append(file)
            continue

        df['Date'] = file_date  # Date
        df['TempMax'] = df['Temperature'].max() # Maximum Temperature
        df['TempMin'] = df['Temperature'].min() # Minimum Temperature
        df['Dew'] = df['Temperature'] - (100 - df['RH']) / 5 # Dew point
        df['PrecpType'] = df['Precp'].apply(lambda x: 1 if x > 0.2 else 0)

        df = df[['Date','TempMax','TempMin',"Temperature","Dew","RH","Precp", "PrecpType","WS","WD","StnPres"]]

        daily_avg_list.append(df)

    daily_avg = pd.concat(daily_avg_list, ignore_index=True)
    # daily_avg['Year'] = daily_avg['Date'].dt.year
    # daily_avg['Month'] = daily_avg['Date'].dt.month
    # daily_avg['Day'] = daily_avg['Date'].dt.day
    # daily_avg = daily_avg.drop(columns=['Date'])
    daily_avg = daily_avg.groupby(['Date']).agg({
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

    daily_avg = daily_avg.round(2)

    # Replace NaN values
    for i, column in enumerate(daily_avg.columns):
        if i == 0:
            continue

        # generates a random number 0 or 1 to decide ffill or bfill
        random_choice = np.random.choice([0, 1])
        if random_choice:
            daily_avg[column] = daily_avg[column].ffill()
        else:
            daily_avg[column] = daily_avg[column].bfill()

    OUTPUT_FOLDER = Path("../data")
    OUTPUT_NAME = Path("data.csv")
    OUTPUT_PATH = OUTPUT_FOLDER / OUTPUT_NAME

    # Check if output folder exist
    if not OUTPUT_FOLDER.is_dir():
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    daily_avg.to_csv(OUTPUT_PATH, index=False)

    print(f"Average data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    data_convert()