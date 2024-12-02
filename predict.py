import argparse
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from mpmath import timing

from data_provider.data_factory import data_provider
from prepare_data import prepare_data
from models import EAGNet


def calculate_data_range(start_date, end_date, seq_len):
    """
    根據預測日期範圍和輸入序列長度，計算數據開始日期
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 計算預測天數
    prediction_days = (end_date - start_date).days + 1

    # 計算所需的數據起始日期
    required_start_date = start_date - timedelta(days=seq_len//4)
    return required_start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), prediction_days


args = argparse.Namespace(
    root_path='./dataset/',                         # 資料集的根目錄
    data_path='predict.csv',                        # 要使用的資料文件名稱
    saved_model='saved_models/EAGNet-v0.2.pth',     # 已經訓練好的模型檔案路徑
    seq_len=80,                                     # 用來預測的序列長度 (輸入序列的天數)
    label_len=0,                                    # 預測用的標籤長度 (通常在預測模式下設為0)
    pred_len=4,                                     # 預測的天數
    enc_in=5,                                       # 模型的輸入特徵數量
    d_model=512,                                    # 模型的隱藏層大小
    dropout=0.5,                                    # Dropout 機率，用於防止過擬合
    features="M",                                   # 特徵模式 ('M' 表示多變量模式)
    target='rain_prob',                             # 預測目標欄位名稱
    start_date="2024-07-21",                        # 預測資料的開始日期
    end_date="2024-07-23",                          # 預測資料的結束日期
    station_name="桃園 (C0C480)",                    # 天氣資料的測站名稱
    output_name="predict",                          # 輸出檔案名稱，用於存儲生成的預測資料
    num_workers=0,
    compare=False,                                   # 與真實值比較
)

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Prepare data
start_date, end_date, pred_days = calculate_data_range(args.start_date, args.end_date, args.seq_len)
if args.compare:
    prepare_data(
        start_date=start_date,
        end_date=end_date,
        station_name=args.station_name,
        output_name=args.output_name
    )
else:
    prepare_data(
        start_date=start_date,
        end_date=(datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"),
        station_name=args.station_name,
        output_name=args.output_name
    )

data, _ = data_provider(args, flag='pred')

# Prepare model
model = EAGNet.Model(args).to(device)
print('loading model\n')
model.load_state_dict(torch.load(args.saved_model))

true = []
pred = []

if args.compare:
    # 確認 data_loader 已正確加載
    with torch.no_grad():
        for i in range(pred_days):
            batch_x, batch_y = data[i*4]

            batch_x = torch.tensor(batch_x).float().to(device).unsqueeze(0)

            pred_y = model(batch_x)

            pred.extend(pred_y[0].squeeze().cpu().numpy())
            true.extend(batch_y[:,-1])

    # 假設 true 和 pred 已經定義
    x = list(range(len(true)))

    # 繪製階梯狀的 true 值
    plt.step(x, true, where='mid', color='blue', label='True (Binary)', linewidth=1.5)

    # 繪製 pred 的連續值
    plt.plot(x, pred, color='red', linewidth=1.5, label='Prediction (Probability)')

    # 添加圖例
    plt.legend()
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title('True (Binary) vs Prediction (Probability)')
    plt.show()

else:
    # 確認 data_loader 已正確加載
    with torch.no_grad():
        for i in range(pred_days):
            batch_x, batch_y = data[i * 4]

            batch_x = torch.tensor(batch_x).float().to(device).unsqueeze(0)

            pred_y = model(batch_x)

            pred.extend(pred_y[0].squeeze().cpu().numpy())

    time = {0: "凌晨", 1: "早上", 2: "下午", 3:"晚上"}
    curr_date = datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=1)
    end_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    for i, p in enumerate(pred):
        if i % 4 == 0:
            print()
            curr_date += timedelta(days=1)

        print(f"{curr_date.strftime('%Y-%m-%d')} {time[i%4]}降雨機率: {p * 100:.2f}%")