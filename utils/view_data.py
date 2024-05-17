import os
import pandas as pd
import matplotlib.pyplot as plt

def view_data(DATA_FOLDER: str="../data"):
    """
    Visualize temperature data from multiple CSV files stored in the specified data folder.

    Args:
        DATA_FOLDER (optional): The folder containing the CSV files to visualize. Defaults to "./data".

    Returns:
        None
    """

    dfs = []

    file_path = os.path.join(DATA_FOLDER, "data.csv")

    # skip first two rows and uses only 2nd column
    df = pd.read_csv(file_path, usecols=[5])    # usecols=[2] if choose to view download directory

    # add download to dataframe
    dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)

    df_combined['Temperature'] = pd.to_numeric(df_combined['Temperature'], errors='coerce')

    # df_combined = df_combined.drop('SeaPres', axis=1)

    df_combined.plot(xlabel='Time', ylabel='Temperature (°C)', figsize=(15, 8))

    print(len(df_combined))

    plt.show()

if __name__ == "__main__":
    view_data()