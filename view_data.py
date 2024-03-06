import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = "./data"  # ./download

dfs = []
for filename in sorted(os.listdir(DATA_FOLDER)):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATA_FOLDER, filename)

        # skip first two rows and uses only 2nd column
        df = pd.read_csv(file_path, usecols=[4])    # usecols=[2] if choose to view download directory

        # add download to dataframe
        dfs.append(df)

df_combined = pd.concat(dfs, ignore_index=True)

df_combined['Temperature'] = pd.to_numeric(df_combined['Temperature'], errors='coerce')

# df_combined = df_combined.drop('SeaPres', axis=1)

df_combined.plot(xlabel='Time', ylabel='Temperature (°C)', figsize=(15, 8))

plt.show()