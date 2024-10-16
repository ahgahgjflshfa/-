import os
import time
import shutil
import numpy as np
import argparse
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

def download_data(
    start_date: str,
    n: int=1,
    driver_path: str | Path=Path("./driver/msedgedriver.exe"),
    dir_name: str="download"
):
    """
    Download data starting from the specified START_DATE for N days.

    Args:
        start_date: The start date from which to download the data in the format 'YYYY-MM-DD'.
        n (optional): The number of days' worth of data to download, counting backwards from START_DATE.
                        Defaults to 1.
        driver_path (optional): The path to the driver executable used for web scraping.
                                Defaults to "driver/msedgedriver.exe".
        dir_name (optional): The directory name of download directory

    Returns:
        None
    """

    MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    START_YEAR = start_date[:4]
    START_MONTH = MONTH_NAMES[int(start_date[5:7]) - 1]
    START_DAY = start_date[8:].lstrip("0")    # delete leading zero

    # Set webdriver path
    service = Service(executable_path=driver_path)

    # Set options
    ie_options = webdriver.EdgeOptions()

    # # Creates download directory if it doesn't exist
    # if not os.path.exists('download'):
    #     os.makedirs('download')

    # Set default download directory
    cur_dir = os.getcwd()
    download_dir = os.path.join(cur_dir, dir_name)
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
    input_element.send_keys("桃園 (C0C480)")

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

    for _ in range(n):
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

# 定義 weather.csv 中的目標特徵
proper_columns = [
    'date', 'StnPres', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)', 'PAR (µmol/m²/s)',
    'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'OT'
]

def process_file(file_path: Path):
    # 提取文件名中的年月日信息
    year, month, day = file_path.name.rstrip(".csv").split("-")[1:]

    # 讀取 CSV 文件
    df = pd.read_csv(file_path, skiprows=1, na_values=['--', '\\', '/', '&', 'X', ' '])

    # 修正 ObsTime 中的 24 為 00 並且日期增加一天
    df['ObsDate'] = f"{year}-{month}-{day}"
    df['ObsTime'] = df['ObsTime'].astype(str)

    # 處理 "24" 的情況，將其替換為 "00" 並將日期加一天
    mask_24 = df['ObsTime'] == "24"
    if mask_24.any():
        df.loc[mask_24, 'ObsTime'] = "00"
        df.loc[mask_24, 'ObsDate'] = pd.to_datetime(df.loc[mask_24, 'ObsDate']) + pd.Timedelta(days=1)

    # 確保 ObsDate 是 datetime 類型，並轉換為字符串格式
    df['ObsDate'] = pd.to_datetime(df['ObsDate']).dt.strftime("%Y-%m-%d")

    # 確保 ObsTime 總是兩位數格式（如 "1" 轉換為 "01"）
    df['ObsTime'] = df['ObsTime'].str.zfill(2)

    # 合併日期和時間，生成完整的日期時間列
    try:
        df['date'] = pd.to_datetime(df['ObsDate'] + " " + df['ObsTime'] + ":00", format="%Y-%m-%d %H:%M")
    except ValueError as e:
        print(f"日期時間解析錯誤: {e}")
        raise

    # 添加溫度相關特徵
    df['T (degC)'] = df['Temperature']
    df['Tpot (K)'] = df['Temperature'] + 273.15  # 假設 Tpot 是絕對溫度
    df['Tdew (degC)'] = df['Temperature'] - (100 - df['RH']) / 5  # 估算露點溫度

    # 相對濕度
    df['rh (%)'] = df['RH']

    # 計算 VPmax, VPact, VPdef
    df['VPmax (mbar)'] = 6.11 * 10**(7.5 * df['Temperature'] / (237.3 + df['Temperature']))
    df['VPact (mbar)'] = df['VPmax (mbar)'] * (df['RH'] / 100)
    df['VPdef (mbar)'] = df['VPmax (mbar)'] - df['VPact (mbar)']

    # 比濕計算（假設為常數換算）
    df['sh (g/kg)'] = 0.622 * df['VPact (mbar)'] / (df['StnPres'] - df['VPact (mbar)']) * 1000

    # 水蒸氣濃度計算
    df['H2OC (mmol/mol)'] = df['sh (g/kg)'] / 18.02 * 1000  # 假設分子量為 18.02

    # 空氣密度
    df['rho (g/m**3)'] = df['StnPres'] / (287.05 * (df['Temperature'] + 273.15))

    # 風速和風向
    df['wv (m/s)'] = df['WS']
    df['max. wv (m/s)'] = df['WS']  # 假設無最大風速數據，用 WS 代替
    df['wd (deg)'] = df['WD']

    # 雨量和下雨狀況
    df['rain (mm)'] = df['Precp']
    df['raining (s)'] = df['Precp'].apply(lambda x: 3600 if x > 0 else 0)  # 假設下雨時為 3600 秒

    # 輻射相關特徵
    df['SWDR (W/m²)'] = df['SunShine'] if 'SunShine' in df.columns else 0  # 使用 SunShine 代替 SWDR
    df['PAR (µmol/m²/s)'] = 0  # 默認設為 0
    df['max. PAR (µmol/m²/s)'] = 0  # 默認設為 0

    # 日誌溫度和 OT
    df['Tlog (degC)'] = df['Temperature']
    df['OT'] = df['Temperature']  # 假設 OT 與溫度相同

    # 只保留目標列
    df = df[proper_columns]

    # 填補缺失值
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].isna().all():
                df[column].fillna(0, inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)

    # 將 StnPres 列名更改為 p (mbar)
    df.rename(columns={'StnPres': 'p (mbar)'}, inplace=True)

    return df.round(2)

def split_data(test_size: float, path: str="download"):
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

def save_to_weather_csv(input_directory, output_file):
    combined_df = pd.DataFrame()
    input_directory = Path(input_directory)
    file_paths = list(input_directory.glob("*.csv"))

    for file_path in file_paths:
        df = process_file(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)

def prepare_data(
    date: str="",
    n: int=1,
    test_size: float=0.2,
    dir_name:str="download",
    split: bool=False,
    output_file="./dataset/weather.csv"
):
    """

    Args:
        date (optional): Starting date. If no value pass, data won't be downloaded.
        n (optional): Number of datas to download. Default is 1.
        test_size (optional): Ratio of train data and test data. Default is 0.2
        dir_name (optional): Directory to put downloaded files. Default is `download`
        split (optional): Split data or not.

    Returns:
        None
    """
    if date:
        download_data(start_date=date, n=n, dir_name=dir_name)

    if split:
        split_data(test_size=test_size)

    save_to_weather_csv(input_directory=dir_name, output_file=output_file)

if __name__ == "__main__":

    prepare_data()