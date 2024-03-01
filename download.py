import os
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

N = 10

# Set webdriver path
service = Service(executable_path='./msedgedriver.exe')

# Set options
ie_options = webdriver.EdgeOptions()

# # Creates data directory if it doesn't exist
# if not os.path.exists('data'):
#     os.makedirs('data')

# Set default download directory
cur_dir = os.getcwd()
download_dir = os.path.join(cur_dir, 'data')
ie_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    })

# Create a driver instance
driver = webdriver.Edge(service=service, options=ie_options)

# goto website
driver.get("https://codis.cwa.gov.tw/StationData")

# select weather station
checkbox_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "auto_C0"))
)
checkbox_element.click()

# choose a specific station area (e.g. "桃園")
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

# click search button
search_button_element = driver.find_element(By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[11]/div/div/div/div[2]')
search_button_element.click()

# click view button
view_button_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/main/div/div/section[1]/div[1]/div[3]/div[1]/div[1]/div[6]/div/div[1]/div/button'))
)
view_button_element.click()

time.sleep(1)

for _ in range(N):
    # download csv to Downloads
    download_button_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[2]/div'))
    )
    download_button_element.click()

    time.sleep(0.5)

    prev_page_element = driver.find_element(By.XPATH, "/html/body/div/main/div/div/section[2]/div/div/section/div[5]/div[1]/div[1]/label/div/div[1]")
    prev_page_element.click()

    time.sleep(2)   # wait for download to complete

driver.quit()