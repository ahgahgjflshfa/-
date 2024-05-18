import os
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class DataIncorrectError(Exception):
    def __init__(self, messages):
        super.__init__(messages)

def download_data(START_DATE: str, N: int=1, driver_path: str | Path="../driver/msedgedriver.exe"):
    """
    Download data starting from the specified START_DATE for N days.

    Args:
        START_DATE: The start date from which to download the data in the format 'YYYY-MM-DD'.
        N (optional): The number of days' worth of data to download, counting backwards from START_DATE.
                        Defaults to 1.
        driver_path (optional):driver_path (str, optional): The path to the driver executable used for web scraping.
                                Defaults to "driver/msedgedriver.exe".

    Returns:
        None
    """

    MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    START_YEAR = START_DATE[:4]
    START_MONTH = MONTH_NAMES[int(START_DATE[5:7]) - 1]
    START_DAY = START_DATE[8:].lstrip("0")    # delete leading zero

    # Set webdriver path
    service = Service(executable_path=driver_path)

    # Set options
    ie_options = webdriver.EdgeOptions()

    # # Creates download directory if it doesn't exist
    # if not os.path.exists('download'):
    #     os.makedirs('download')

    # Set default download directory
    cur_dir = os.getcwd()
    parent_dir = os.path.dirname(cur_dir)
    download_dir = os.path.join(parent_dir, 'download')
    ie_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        })

    # Create a driver instance
    driver = webdriver.Edge(service=service, options=ie_options)

    # goto website
    driver.get("https://codis.cwa.gov.tw/StationData")

    # select temperature station
    checkbox_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "auto_C0"))
    )
    checkbox_element.click()

    # choose a specific area (e.g. "桃園")
    select_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "station_area"))
    )
    select = Select(select_element)
    select.select_by_index(4)

    # enter specific station name (e.g. "中壢 (C0C700)")
    input_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div/main/div/div/div/div/aside/div/div[1]/div/div/section/ul/li[5]/div/div[2]/div/input"))
    )
    input_element.send_keys("中壢 (C0C700)")

    time.sleep(1)

    # click station icon
    search_button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[11]/div/div/div/div[2]')
    search_button_element.click()

    # click view button
    view_button_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[6]/div/div[1]/div/button'))
    )
    view_button_element.click()

    time.sleep(1)

    # start from specific date
    date_button_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/input'))
    )
    date_button_element.click()

    # find year select button and click it
    year_select_button_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/div/div[2]/div[1]/div[1]'))
    )
    year_select_button_element.click()

    # select specific year (e.g. "2023")
    xpath = f'//div[contains(@class, "vdatetime-year-picker__item") and contains(text(), "{START_YEAR}")]'
    year_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    year_element.click()

    # find month select button and click it
    month_select_button_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/div/div[2]/div[1]/div[2]'))
    )
    month_select_button_element.click()

    # select specific month (e.g. "04")
    xpath = f'//div[contains(@class, "vdatetime-month-picker__item") and contains(text(), "{START_MONTH}")]'
    month_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    month_element.click()

    # select specific day (e.g. "17")
    xpath = f'//div[contains(@class, "vdatetime-calendar__month__day") and descendant::span/span[text()="{START_DAY}"]]'
    day_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, xpath))
    )
    day_element.click()

    for _ in range(N):
        # download csv to Downloads
        download_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div'))
        )
        download_button_element.click()

        time.sleep(0.5)

        prev_page_element = driver.find_element(By.XPATH, "/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[1]")
        prev_page_element.click()

        time.sleep(1)   # wait for download to complete

    driver.quit()

def process_file(file_path: Path):
    year, month, day = file_path.name.rstrip(".csv").split("-")[1:]

    proper_columns = ['ObsTime', "Temperature", "Dew", "RH", "Precp", "WS", "WD", "StnPres",
             'Month_01', 'Month_02', 'Month_03', 'Month_04', 'Month_05', 'Month_06', 'Month_07', 'Month_08',
             'Month_09', 'Month_10', 'Month_11', 'Month_12', 'PrecpType_None', 'PrecpType_Rain']

    # Read csv file
    df = pd.read_csv(file_path, skiprows=1, na_values=['--', '\\', '/', '&', 'X', ' '])  # fucking stupid

    df['Dew'] = df['Temperature'] - (100 - df['RH']) / 5  # Dew point
    df['Month'] = month  # Date
    df['PrecpType'] = df['Precp'].apply(lambda x: 1 if x > 0.2 else 0)

    month_dummies = ['Month_01', 'Month_02', 'Month_03', 'Month_04', 'Month_05', 'Month_06', 'Month_07', 'Month_08',
               'Month_09', 'Month_10', 'Month_11', 'Month_12']

    for m, dummy in enumerate(month_dummies):
        df[dummy] = df['Month'].apply(lambda x: True if int(x) == m + 1 else False)

    precptype_dummies = ['PrecpType_None', 'PrecpType_Rain']

    for t, dummy in enumerate(precptype_dummies):
        df[dummy] = df['PrecpType'].apply(lambda x: True if x == t else False)

    df = df[proper_columns]

    # Fill NaN values with mean, forward data, or backward data
    for i, column in enumerate(df.columns):
        # Check if the data downloaded aren't the right data
        if column not in proper_columns:
            raise DataIncorrectError(f"Should not have data column {column}")

        # Fill entire column NaNs with mean or 0 if the entire column is NaN
        if df[column].isna().all():
            df[column].fillna(0, inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    return (year, month, day), df.round(2)

def split_data(test_size: float, path: str | Path = Path("download")):
    data_path = Path(path)
    train_dir = Path('data/train')
    test_dir = Path('data/test')

    if train_dir.exists():
        shutil.rmtree(train_dir)

    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Create train and test directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # List all CSV files in the data directory
    csv_files = list(data_path.glob("*.csv"))

    train_files, test_files = train_test_split(csv_files, test_size=test_size, shuffle=True)

    for file in train_files:
        try:
            (year, month, day), df = process_file(file)
        except pd.errors.EmptyDataError or DataIncorrectError as e:
            print(f"File {file} is empty or incorrect: {e}")
            continue

        # Check if file already exists
        if (train_dir / f"{year}-{month}-{day}.csv").exists():
            (train_dir / f"{year}-{month}-{day}.csv").unlink()

        # Write file
        df.to_csv(train_dir / f"{year}-{month}-{day}.csv", index=False)

    for file in test_files:
        try:
            (year, month, day), df = process_file(file)
        except pd.errors.EmptyDataError as _:
            print(f"File {file} is empty or incorrect.")
            continue

        # Check if file already exists
        if (test_dir / f"{year}-{month}-{day}.csv").exists():
            (test_dir / f"{year}-{month}-{day}.csv").unlink()

        # Write file
        df.to_csv(test_dir / f"{year}-{month}-{day}.csv", index=False)

    print(f'Saved {len(train_files)} files to {train_dir}')
    print(f'Saved {len(test_files)} files to {test_dir}')

def prepare_data(date: str="", n: int=1, test_size: float=0.2):
    """

    Args:
        date: Starting date.
        n: Number of datas to download.

    Returns:
        None
    """
    if date:
        download_data(date, n)

    split_data(test_size=test_size)

if __name__ == "__main__":
    prepare_data()