import torch
import argparse
from exp.exp_main import Exp_Main

def main():
    args = argparse.Namespace(
        model_id="weather_80",
        model="EAGNet",
        data="custom",
        root_path="./dataset/",
        data_path="桃園 (C0C480).csv",
        features="M",
        target="rain_prob",                     # 这里你需要确保目标列存在，并且改成你需要预测的特征
        checkpoints="./checkpoints/",
        seq_len=80,                             # 用前幾個時間段的資料作為輸入
        label_len=0,
        pred_len=4,                             # 預測後面多少個時間段
        seg_len=20,                             # For SegRNN (要跟著 pred_len 一起改動)
        enc_in=5,                               # 輸入特徵數量
        enc_out=1,                              # For SegRNN (輸出特徵數量)
        d_model=512,                            # 隱藏層大小
        dropout=0.5,
        num_workers=10,
        train_epochs=30,
        batch_size=64,
        patience=10,
        lr=0.1,
        loss="bce",
        lradj="type3",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0,1",
        test_flop=False,
        itr=1,
        rnn_type="gru",
        dec_way="pmf",
        des="test",
    )

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.dropout,
        args.rnn_type,
        args.dec_way,
        args.seg_len,
        args.loss,
        args.des,
        ii
    )

    exp = Exp(args)  # 設定實驗
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()