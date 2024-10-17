# Project: Weather Prediction Model

## Introduction

This project focuses on using time-series deep learning models for weather forecasting. Specifically, it aims to predict key meteorological parameters such as temperature and rainfall. The project is based on and extends the SegRNN model, which serves as the core reference for this work due to its impressive capabilities in handling time-series data.

The implementation is done using Python and PyTorch, with a custom dataset used for training and validation. The ultimate goal is to train an accurate multi-output prediction model capable of forecasting multiple weather variables simultaneously.

## Objectives

- Build a deep learning-based weather forecasting model.
- Utilize LSTM and SegRNN for time-series prediction, focusing on the accuracy of temperature and rainfall predictions.
- Explore the features of SegRNN and adapt it to suit meteorological data requirements.

## Key Features

- **Multi-output Prediction**: The model predicts multiple weather indicators, such as temperature and rainfall, simultaneously.
- **Time Encoding**: Time features like month, day, weekday, and hour are incorporated into the input data to enhance prediction accuracy.
- **Flexible Dataset**: The project supports custom datasets, allowing for flexible data loading and expansion, with built-in standardized data preprocessing.

## File Structure

- `data_provider/`: Contains logic for data loading and processing.
  - `data_loader.py`: Defines custom datasets such as Dataset_Custom.
  - `data_factory.py`: Provides methods to create datasets, supporting different types of data.
- `exp/`: Contains training, validation, and testing logic.
  - `exp_main.py`: Defines the model's training process, validation, and testing procedures.
- `models/`: Contains model definitions.
  - `SegRNN.py`: Defines the SegRNN model structure, used for time-series forecasting.
- `utils/`: Contains utility functions.
  - `metrics.py`: Computes evaluation metrics such as MAE, MSE, etc.
  - `tools.py`: Includes helper functions for model training.

## Usage

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Required packages can be installed using:
  ```sh
  pip install -r requirements.txt
  ```

### Running the Model

1. **Training**: To train the model, use the following command:
   ```sh
   python run_longExp.py --config configs/weather_config.yaml
   ```
   Modify the configuration file to set the appropriate parameters for training.

2. **Testing**: After training, you can evaluate the model using:
   ```sh
   python run_longExp.py --test --config configs/weather_config.yaml
   ```

3. **Prediction**: For making predictions, use the `predict` method defined in `exp_main.py`.

### Parameters

- `seq_len`: Length of the input sequence.
- `pred_len`: Length of the prediction sequence.
- `features`: Type of features used (e.g., 'S' for single feature, 'M' for multiple features).
- `targets`: List of targets to predict (e.g., temperature, rainfall).

## Reference

This project references and extends the SegRNN model. For more information, please refer to the original [SegRNN paper](https://arxiv.org/abs/2105.03906).

## Acknowledgments

- [**SegRNN**](https://github.com/lss-1138/SegRNN): The core reference for the model architecture, which significantly influenced the development of this project.

