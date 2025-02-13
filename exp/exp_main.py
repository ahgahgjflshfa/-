import torchmetrics.classification

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LSTM, SegRNN, EAGNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torchmetrics.classification import BinaryAccuracy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


# 自定義損失函數，結合Brier Score和Focal Loss
class WeatherLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=1):
        super(WeatherLoss, self).__init__()
        self.alpha = alpha  # 正例權重
        self.beta = beta  # 平滑係數

    def forward(self, pred, target):
        # 標籤平滑化處理，避免模型過於自信
        smoothed_targets = target * (1 - self.beta) + 0.5 * self.beta

        # 計算加權的BCE loss
        bce = -(smoothed_targets * torch.log(pred + 1e-7) +
                (1 - smoothed_targets) * torch.log(1 - pred + 1e-7))

        # 對正例施加更大權重
        weights = torch.where(target > 0.5, self.alpha, 1.0)
        weighted_bce = weights * bce

        # 添加KL散度作為正則化項，防止預測過於極端
        kl_div = pred * torch.log(pred / (smoothed_targets + 1e-7) + 1e-7) + \
                 (1 - pred) * torch.log((1 - pred) / (1 - smoothed_targets + 1e-7) + 1e-7)

        # 最終loss：加權BCE + KL散度正則化
        loss = weighted_bce.mean() + 0.1 * kl_div.mean()

        return loss


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'EAGNet': EAGNet,
            'SegRNN': SegRNN,
            'LSTM': LSTM
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)

        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()

        elif self.args.loss == "mse":
            criterion = nn.MSELoss()

        elif self.args.loss == "bce":
            criterion = nn.BCELoss()

        elif self.args.loss == "custom":
            criterion = WeatherLoss()

        else:
            criterion = nn.L1Loss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_acc = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                accuracy_metric = BinaryAccuracy()
                acc = accuracy_metric(pred, true)

                total_loss.append(loss)
                total_acc.append(acc)

        total_loss = np.average(total_loss)
        total_acc = np.average(total_acc)
        self.model.train()

        return total_loss, total_acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # if self.args.load_before_train:
        #     print('loading model\n')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.ReduceLROnPlateau(
            model_optim, mode='min', factor=0.5, patience=10, verbose=True
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                else:
                    outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # print(nn.Sigmoid(batch_y))
                    #
                    # print(outputs)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s\n'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()

                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print(f"Accuracy | Vali accuracy: {vali_acc:.2f} Test accuracy: {test_acc:.2f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model\n')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs # test_data.inverse_transform(outputs)  # .squeeze()
                true = batch_y # test_data.inverse_transform(batch_y)  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 5 == 0:
                    input = batch_x.detach().cpu().numpy()

                    pred = outputs[0]
                    true = batch_y[0]
                    input = input[0]

                    gt = np.concatenate((input[:, -1], true[:, -1]), axis=0)
                    pd = np.concatenate((input[:, -1], pred[:, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        results = metric(preds, trues)
        print(f'Brier Score: {results["Brier Score"]:.2f}, ROC AUC: {results["ROC AUC"]:.2f}, '
              f'Precision: {results["Precision"]:.2f}, Recall: {results["Recall"]:.2f}')
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'Brier Score: {results["Brier Score"]:.2f}, ROC AUC: {results["ROC AUC"]:.2f}, '
              f'Precision: {results["Precision"]:.2f}, Recall: {results["Recall"]:.2f}')
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # Plot all predictions vs actual values as a line plot
        plt.figure(figsize=(12, 6))
        plt.plot(trues.reshape(-1), label='True Values', color='blue', alpha=0.3)
        plt.plot(preds.reshape(-1), label='Predictions', color='red')
        plt.title(f'{setting} - Predictions vs True Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        save_name = self.args.data_path.split()[0]
        plt.savefig(os.path.join(folder_path, f"{save_name}.png"))

        plt.legend()
        plt.show()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)   # Scaled

                        original_prediction = outputs

                else:
                    outputs = self.model(batch_x)   # Scaled

                    original_prediction = outputs

                pred = original_prediction.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return