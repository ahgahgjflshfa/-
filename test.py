import torch
import argparse
from exp.exp_main import Exp_Main

def main():
    args = argparse.Namespace(
        random_seed=2024,
        is_training=True,  # 記得要改
        model_id="weather_80",
        model="EAGNet",
        data="custom",
        root_path="./dataset/",
        data_path="桃園 (C0C480).csv",
        features="M",
        target="rain_prob",  # 这里你需要确保目标列存在，并且改成你需要预测的特征
        freq="h",
        checkpoints="./checkpoints/",
        scaler="./scaler/",
        seq_len=80,
        label_len=0,
        pred_len=4,
        seg_len=20,  # 要跟著 pred_len 一起改動
        enc_in=5,  # 修改为你的特征数量?
        enc_out=1,  # 輸出
        d_model=512,
        dropout=0.5,
        do_predict=False,
        num_workers=10,
        train_epochs=30,
        batch_size=64,
        patience=10,
        lr=0.1,
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