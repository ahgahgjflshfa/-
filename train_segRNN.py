import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

# 設定實驗參數
def main():
    args = argparse.Namespace(
        random_seed=2024,
        is_training=True,       # 記得要改
        model_id="weather_720",
        model="SegRNN",
        data="custom",
        root_path="./dataset/",
        data_path="weather.csv",
        features="M",
        target="rain_prob",        # 这里你需要确保目标列存在，并且改成你需要预测的特征
        freq="h",
        checkpoints="./checkpoints/",
        seq_len=120,
        label_len=0,
        pred_len=4,
        seg_len=2,         # 要跟著 pred_len 一起改動
        enc_in=4,          # 修改为你的特征数量?
        enc_out=1,          # 輸出
        d_model=512,
        dropout=0.5,
        do_predict=False,
        num_workers=10,
        train_epochs=30,
        batch_size=64,
        patience=10,
        learning_rate=0.0001,
        loss="bce",
        lradj="type3",
        pct_start=0.3,
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

    # 設定隨機種子
    # fix_seed = args.random_seed
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # 設定實驗記錄
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
                args.des, ii)

            exp = Exp(args)  # 設定實驗
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
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