import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 獲取配置參數
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.output_features = 2  # 預測溫度和降雨量
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # 定義 LSTM 層
        self.lstm1 = nn.LSTM(
            input_size=self.enc_in,  # 根據 configs.enc_in
            hidden_size=self.d_model,
            num_layers=1,  # 可以根據需求調整
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True
        )

        # 定義全連接輸出層
        self.dense = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len * self.output_features)  # 與第二個模型相同
        )

    def forward(self, x):
        # 初始化隱藏狀態和記憶狀態
        h0 = torch.zeros(1, x.size(0), self.d_model).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.d_model).to(x.device)

        # 正向傳播
        x1, _ = self.lstm1(x, (h0.detach(), c0.detach()))
        x2, _ = self.lstm2(x1, (h0.detach(), c0.detach()))
        out, _ = self.lstm3(x2, (h0.detach(), c0.detach()))

        # 全連接層處理
        out = self.dense(out[:, -1, :])

        # Reshape 成 [batch_size, pred_len, enc_in]
        out = out.view(-1, self.seg_num_y * self.seg_len, self.output_features)
        return out
