import os
import time
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

def split_data(path: str | Path = Path("../data/data.csv")):
    train_data, test_data = train_test_split(pd.read_csv(path), test_size=0.2)

    train_data.to_csv(Path("../data/train.csv"), index=False)
    test_data.to_csv(Path("../data/test.csv"), index=False)

def prepare_data(date: str, n: int=1):
    """

    Args:
        date: Starting date.
        n: Number of datas to download.

    Returns:
        None
    """
    download_data(date, n)
    data_convert()
    split_data()

if __name__ == "__main__":
    prepare_data("2024-04-17", 1)