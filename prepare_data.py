import os
import time
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta

class DataIncorrectError(Exception):
    def __init__(self, messages):
        super.__init__(messages)

def download_data(
    start_date: str,
    end_date: str,
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

    # Convert start_date and end_date to datetime objects
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError("The start date cannot be after the end date.")

    START_YEAR = start_date[:4]
    START_MONTH = MONTH_NAMES[int(start_date[5:7]) - 1]
    START_DAY = start_date[8:].lstrip("0")    # delete leading zero

    # Set webdriver path
    service = Service(executable_path=driver_path)

    # Set options
    ie_options = webdriver.EdgeOptions()

    # Enable headless mode
    ie_options.add_argument("headless")
    ie_options.add_argument("disable-gpu")

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

    # Iterate over the date range
    current_date = start_dt

    while current_date <= end_dt:
        # download csv to Downloads
        download_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div'))
        )
        download_button_element.click()

        # time.sleep(0.5)   # wait for download to complete

        next_page_element = driver.find_element(By.XPATH, "/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[4]")
        next_page_element.click()

        # Move to the next date
        current_date += timedelta(days=1)

        time.sleep(0.5)

    driver.quit()


# 定義 weather.csv 中的目標特徵
proper_columns = [
    'date', 'longitude', 'latitude', 'month', 'season', 'StnPres', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'wv (m/s)', 'wd (deg)',
    'rain (mm)'
]


def process_file(file_path: Path):
    try:
        year, month, day = file_path.name.rstrip(".csv").split("-")[1:]

        # 讀取 CSV 文件
        df = pd.read_csv(file_path, skiprows=1, na_values=['--', '\\', '/', '&', 'X', ' '])

        # 新增經緯度欄位，並填入固定的經緯度值
        df['longitude'] = 121.3232
        df['latitude'] = 24.9924

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
        df['date'] = pd.to_datetime(df['ObsDate'] + " " + df['ObsTime'] + ":00", format="%Y-%m-%d %H:%M")

        # 新增月份欄位
        df['month'] = df['date'].dt.month

        # 新增季節欄位
        df['season'] = df['date'].dt.month % 12 // 3 + 1

    except ValueError:
        # 讀取 CSV 文件
        df = pd.read_csv(file_path, skiprows=0, na_values=['--', '\\', '/', '&', 'X', ' '])

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

    # 風速和風向
    df['wv (m/s)'] = df['WS']
    df['wd (deg)'] = df['WD']

    # 雨量和下雨狀況
    df['rain (mm)'] = df['Precp']

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



def save_to_weather_csv(input_directory, output_file, start_date=None, end_date=None):
    combined_df = pd.DataFrame()
    input_directory = Path(input_directory)
    file_paths = list(input_directory.glob("*.csv"))

    # 如果 start_date 和 end_date 不為空，將它們轉換為 datetime 格式
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    for file_path in file_paths:
        # 提取檔案名中的日期部分，例如 'C0C480-2009-06-22.csv' 中的 '2009-06-22'
        date_parts = file_path.stem.split('-')[-3:]  # 提取['2009', '06', '22']
        file_date_str = '-'.join(date_parts)  # 合併成 '2009-06-22'
        file_date = pd.to_datetime(file_date_str, format='%Y-%m-%d')  # 轉換為 datetime

        # 日期篩選邏輯：僅在 start_date 或 end_date 存在的情況下進行過濾
        if (start_date and file_date < start_date) or (end_date and file_date > end_date):
            continue  # 跳過不在範圍內的檔案

        # 處理符合條件的檔案
        df = process_file(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 將篩選後的資料保存為 CSV
    combined_df.to_csv(output_file, index=False)


def prepare_data(
    download: bool=False,
    start_date: str= "",
    end_date: str="",
    dir_name:str="download",
    output_file="./dataset/weather.csv"
):
    """

    Args:
        download (optional): Download or not.
        start_date (optional): Starting date. If no value pass, data won't be downloaded.
        end_date (optional): Ending date.
        dir_name (optional): Directory to put downloaded files. Default is `download`.
        output_file (optional): Where to save output data. Default is `dataset/weather.csv`

    Returns:
        None

    Outputs:
        How many features in dataset.
    """
    if download:
        download_data(start_date=start_date, end_date=end_date, dir_name=dir_name)

    save_to_weather_csv(input_directory=dir_name, output_file=output_file, start_date=start_date, end_date=end_date)

    print(f"{len(proper_columns)} features.")

if __name__ == "__main__":

    prepare_data()