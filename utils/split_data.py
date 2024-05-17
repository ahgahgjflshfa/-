import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_data(path: str | Path = Path("../data/data.csv")):
    train_data, test_data = train_test_split(pd.read_csv(path), test_size=0.2)

    train_data.to_csv(Path("../data/train.csv"), index=False)
    test_data.to_csv(Path("../data/test.csv"), index=False)

if __name__ == "__main__":
    split_data()