# 專題
AI專題 - model to predict future weather using LSTM model.

Data from [here](https://codis.cwa.gov.tw/StationData).

Get webdriver from [here](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/?form=MA13LH).

# Guide
## Prerequisites
1. **Install Python Packages**:  
Ensure you have the necessary Python packages installed. You can install them using the `requirements.txt` file:  
```shell
pip install -r requirements.txt
```

2. **Install Webdriver**:  
Download the appropriate Edge WebDriver, downloaded driver can be placed in the `driver/` directory.

### Notes
* Ensure driver_path points to the correct location of the WebDriver executable.
* The script will create data/train and data/test directories to store the split data.
* The script handles exceptions for empty or incorrect data files during processing.
* Adjust the sleep times (time.sleep()) as needed to accommodate different network speeds or page load times.
  (Through editing `prepare_data.py` file).
* To check for NaN values in your data files, you can use the `test_data.py` script located in the `utils` directory. 
This script provides a convenient way to inspect your data files in `data/train` and `data/test` for any NaN values, 
ensuring the integrity of your dataset.


## Model usage guide
### Dataset Preparation
#### Downloading Data
The prepare_data.py script downloads and processes weather data. To use this script, follow these steps:  
1. Ensure you have the necessary WebDriver installed.
2. Run the script with the desired parameters.
    ```shell
    python prepare_data.py [--date DATE] [--n N] [--test_size TEST_SIZE] [--dir_name DIR_NAME] [--split SPLIT]
    ```
   * `--date DATE`: Starting date. Data will be download backward. If no value is passed, data won't be downloaded. (default: "")
   * `--n N`: Number of data to download. Default is 1. (default: 1)
   * `--test_size TEST_SIZE`: Ratio of test data size. Default is 0.2. (default: 0.2)
   * `--dir_name DIR_NAME`: Directory to put downloaded files. Default is download. (default: "download")
   * `--split SPLIT`: Whether to split the data or not. (default: True)
   
### Training Model
#### Training Script
Use the `train_model.py` script to train your model.

### Evaluating the Model
After training, you can evaluate your model using functions in `utils/plot_function`.

# References:
* [Time series prediction — LSTM的各種用法](https://peaceful0907.medium.com/time-series-prediction-lstm%E7%9A%84%E5%90%84%E7%A8%AE%E7%94%A8%E6%B3%95-ed36f0370204)
* [Time-series Forecasting using LSTM (PyTorch implementation)](https://medium.com/@ozdogar/time-series-forecasting-using-lstm-pytorch-implementation-86169d74942e)