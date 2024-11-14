import os
import time
import pandas as pd
from pathlib import Path

from numpy.f2py.auxfuncs import throw_error
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta

import warnings


class DataIncorrectError(Exception):
    def __init__(self, messages):
        super.__init__(messages)

def download_data(
    start_date: str = None,
    end_date: str = None,
    date_list: list[str] = None,  # 選擇多個特定日期
    driver_path: str | Path = Path("./driver/msedgedriver.exe"),
    station_name: str = "桃園 (C0C480)",
    dir_name: str="download"
):
    """
    Download data for the specified date range or list of dates.

    Args:
        start_date (optional): Start date in 'YYYY-MM-DD' format. Used if date_list is None.
        end_date (optional): End date in 'YYYY-MM-DD' format. Used if date_list is None.
        date_list (optional): Specific list of dates in 'YYYY-MM-DD' format.
        driver_path (optional): Path to the driver executable for web scraping.
        station_name (optional): Name of the weather station.
        dir_name (optional): Directory name for downloaded files.

    Returns:
        None
    """

    MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # 檢查並生成日期清單
    if date_list is None:
        if start_date and end_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError("The start date cannot be after the end date.")
            # 生成範圍內的日期清單
            date_list = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in
                         range((end_dt - start_dt).days + 1)]
        else:
            raise ValueError("Either date_list or start and end dates must be provided.")

    # # Convert start_date and end_date to datetime objects
    # start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    # end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    #
    # if start_dt > end_dt:
    #     raise ValueError("The start date cannot be after the end date.")
    #
    # START_YEAR = start_date[:4]
    # START_MONTH = MONTH_NAMES[int(start_date[5:7]) - 1]
    # START_DAY = start_date[8:].lstrip("0")    # delete leading zero

    # Set webdriver path
    service = Service(executable_path=driver_path)

    # Set options
    ie_options = webdriver.EdgeOptions()

    # Enable headless mode
    # ie_options.add_argument("headless")
    # ie_options.add_argument("disable-gpu")

    # Set default download directory
    cur_dir = os.getcwd()
    download_dir = os.path.join(cur_dir, dir_name).replace('/', '\\')

    ie_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
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

    # enter specific station name (e.g. "中壢 (C0C700)")
    input_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div/main/div/div/div/div/aside/div/div[1]/div/div/section/ul/li[5]/div/div[2]/div/input"))
    )
    input_element.send_keys(station_name)

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

    for date_str in date_list:
        # Parse date into components
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year, month, day = date_obj.year, MONTH_NAMES[date_obj.month - 1], date_obj.day

        # Click on date select button
        date_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/div[1]/input'))
        )
        date_button_element.click()

        year_select_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/div[1]/div/div[2]/div[1]/div[1]'))
        )
        year_select_button_element.click()

        # select specific year (e.g. "2023")
        xpath = f'//div[contains(@class, "vdatetime-year-picker__item") and contains(text(), "{year}")]'
        year_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        year_element.click()

        # find month select button and click it
        month_select_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[2]/div[1]/div/div[2]/div[1]/div[2]'))
        )
        month_select_button_element.click()

        # select specific month (e.g. "04")
        xpath = f'//div[contains(@class, "vdatetime-month-picker__item") and contains(text(), "{month}")]'
        month_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        month_element.click()

        # select specific day (e.g. "17")
        xpath = f'//div[contains(@class, "vdatetime-calendar__month__day") and descendant::span/span[text()="{day}"]]'
        day_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        day_element.click()

        time.sleep(0.5)

        # download csv to Downloads
        download_button_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div'))
        )
        download_button_element.click()

    time.sleep(1)

    driver.quit()


def process_file(file_path: Path):
    try:
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
        df['date'] = pd.to_datetime(df['ObsDate'] + " " + df['ObsTime'] + ":00", format="%Y-%m-%d %H:%M")

    except ValueError:
        # 讀取 CSV 文件
        df = pd.read_csv(file_path, skiprows=0, na_values=['--', '\\', '/', '&', 'X', ' '])

    try:
        # 添加所需的特徵
        df['T (degC)'] = df['Temperature']  # 溫度
        df['rh (%)'] = df['RH']  # 相對濕度
        df['wd (deg)'] = df['WD']  # 風向
        df['p (mbar)'] = df['StnPres']  # 氣壓
        df['rain (mm)'] = df['Precp']  # 雨量
    except KeyError:
        pass

    # 只保留所需的特徵
    try:
        df = df[['date', 'p (mbar)', 'T (degC)', 'wd (deg)', 'rh (%)', 'rain (mm)']]

    except KeyError as e:
        raise KeyError(repr(e) + f" in file {file_path}")

    # 將時間按 6 小時分組
    df['6h_period'] = (df['date'].dt.hour // 6) % 4  # 將每個日期時間按 6 小時的間隔分組

    # 將 6h_period 映射為易於理解的時間段標示，並處理 00:00 的跨日問題
    period_labels = {0: '深夜', 1: '早上', 2: '下午', 3: '晚上'}
    df['6h_period'] = df['6h_period'].map(period_labels)

    # 調整 00:00 時間屬於前一天的「深夜」
    midnight_mask = (df['date'].dt.hour == 0)
    df.loc[midnight_mask, '6h_period'] = '深夜'
    df.loc[midnight_mask, 'date'] = df.loc[midnight_mask, 'date'] - pd.Timedelta(days=1)

    # 創建只保留年月日的標示
    df['day'] = df['date'].dt.date

    grouped_df = df

    # 按日期和 6 小時時間段進行分組，計算每個時間段的平均值（除了雨量是總和）
    grouped_df = df.groupby(['day', '6h_period']).agg({
        'p (mbar)': 'mean',
        'T (degC)': 'mean',
        'wd (deg)': 'mean',
        'rh (%)': 'mean',
        'rain (mm)': 'sum'  # 雨量累積
    }).reset_index()

    # 設定正確的時間段排序順序
    period_order = ['深夜', '早上', '下午', '晚上']
    grouped_df['6h_period'] = pd.Categorical(grouped_df['6h_period'], categories=period_order, ordered=True)

    # 排序結果
    grouped_df = grouped_df.sort_values(by=['day', '6h_period'])

    # 填補缺失值
    for column in grouped_df.columns:
        if pd.api.types.is_numeric_dtype(grouped_df[column]):
            if grouped_df[column].isna().all():
                grouped_df[column] = grouped_df[column].fillna(0)
            else:
                grouped_df[column] = grouped_df[column].fillna(grouped_df[column].mean())

    # 四捨五入保留兩位小數
    grouped_df = grouped_df.round(2)

    # 計算雨量的概率（雨量 >= 1 mm 視為降雨）
    grouped_df['rain_prob'] = (grouped_df['rain (mm)'] >= 1).astype(int)

    grouped_df = grouped_df.drop(["day"], axis=1)

    # 最終結果
    grouped_df = grouped_df.reset_index(drop=True)

    return grouped_df


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

        with warnings.catch_warnings(record=True) as w:
            # 處理符合條件的檔案
            df = process_file(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

            # 確認警告是否出現
            if w and issubclass(w[-1].category, FutureWarning):
                print(f"捕捉到 FutureWarning！請檢查檔案: {file_path}")

    # 將篩選後的資料保存為 CSV
    combined_df.to_csv(output_file, index=False)


def prepare_data(
    station_name,
    download: bool=False,
    start_date: str= None,
    end_date: str = None,
    date_list: list[str] = None,
    download_dir:str= "download",
    output_dir="./dataset",
    output_name=None,
    combine=True
):
    """
    A function to prepare data for model.

    Args:
        download (optional): Download or not.
        start_date (optional): Starting date. If no value pass, data won't be downloaded.
        end_date (optional): Ending date.
        date_list (optional): Specify specific dates' data to be downloaded.
        download_dir (optional): Directory to put downloaded files. Default is `download`.
        station_name (optional): Which weather station's data to download.
        output_dir (optional): Where to save output data. Default is `dataset/weather.csv`
        combine (optional): Combine data into one csv file or not. Default is True.

    Returns:
        None

    Outputs:
        How many features in dataset.
    """
    if download:
        download_data(
            start_date=start_date,
            end_date=end_date,
            date_list=date_list,
            station_name=station_name,
            dir_name=f"{download_dir}/{station_name}"
        )

    if combine:
        if not output_name:
            save_to_weather_csv(
                input_directory=f"{download_dir}/{station_name}",
                output_file=f"{output_dir}/{station_name}.csv",
                start_date=start_date,
                end_date=end_date
            )
        else:
            save_to_weather_csv(
                input_directory=f"{download_dir}/{station_name}",
                output_file=f"{output_dir}/{output_name}.csv",
                start_date=start_date,
                end_date=end_date
            )

    # print(f"{len(proper_columns)} features.")

if __name__ == "__main__":
    date_list = ['2024-07-02', '2024-07-11', '2024-07-21']

    prepare_data(download=False, start_date="2024-07-01", end_date="2024-07-22", station_name="富源 (C0Z080)", output_name="predict")