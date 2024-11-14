import pandas as pd
import numpy as np
import os
import joblib

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4

        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Prepare scaler
        if self.set_type != 0:
            if os.path.exists('./scaler/scaler.joblib'):
                self.scaler = joblib.load('./scaler/scaler.joblib')
            else:
                raise ValueError("No scaler found! Please make sure you have trained the model and saved the scaler.")

        # Read data
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # categorical_columns = ['climate']
        # if categorical_columns:
        #     df_raw = pd.get_dummies(df_raw, columns=categorical_columns)

        '''
        df_raw.columns: ['6h_period', ...(other features), 'rain (mm)', 'rain_prob']
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('6h_period')
        df_raw = df_raw[['6h_period'] + cols + [self.target]]       # Rearrange columns

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] # train, vali, test 起點
        border2s = [num_train, num_train + num_vali, len(df_raw)]                       # train, vali, test 終點

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]  # 去掉date
            df_data = df_raw[cols_data]

        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            feature_data = df_data.iloc[:, :-1]
            true_label = df_data.iloc[:, -1:]

            # Fit
            if self.set_type == 0:  # if train
                train_data = feature_data[border1s[0]:border2s[0]]  # fit train data
                self.scaler.fit(train_data.values)

                # Save scaler
                if not os.path.exists('./scaler/'):
                    os.makedirs('./scaler')

                joblib.dump(self.scaler, './scaler/scaler.joblib')

            scaled_features = self.scaler.transform(feature_data.values)

            data = np.hstack([scaled_features, true_label.values.reshape(-1, 1)])

        else:
            data = df_data.values

        self.data_x = data[border1:border2][:, 1:]  # features (scaled)
        self.data_y = data[border1:border2][:, -1:] # rain true label

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Load scaler
        if os.path.exists('./scaler/scaler.joblib'):
            self.scaler = joblib.load('./scaler/scaler.joblib')
        else:
            raise ValueError("No scaler found! Please make sure you have trained the model and saved the scaler.")

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['6h_period', ...(other features), 'rain (mm)', 'rain_prob']
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)

        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('6h_period')

        df_raw = df_raw[['6h_period'] + cols + [self.target]]
        border1 = 0
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            feature_data = df_data.iloc[:, :-1]  # ignore rain
            true_label = df_data.iloc[:, -1:]

            scaled_features = self.scaler.transform(feature_data.values)

            data = np.hstack([scaled_features, true_label.values.reshape(-1, 1)])

        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]

        else:
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
