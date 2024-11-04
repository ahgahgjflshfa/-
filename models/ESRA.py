import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        hidden_size = 64  # 減少隱藏層大小，使模型更簡單

        # 1. 簡單的特徵處理層
        self.input_layer = nn.Linear(self.enc_in, hidden_size)

        # 2. LSTM層 - 使用單層LSTM替代複雜的GRU+Attention結構
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # 3. 預測層
        self.predict = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加少量dropout防止過擬合
            nn.Linear(32, self.pred_len),
            nn.Sigmoid()
        )

        # 4. 批標準化 - 幫助穩定訓練
        self.batch_norm = nn.BatchNorm1d(self.seq_len)

    def forward(self, x):
        # x shape: [batch, seq_len, features]

        # 1. 批標準化
        x = self.batch_norm(x)

        # 2. 特徵映射
        x = self.input_layer(x)  # [batch, seq_len, hidden_size]

        # 3. LSTM處理時序信息
        lstm_out, _ = self.lstm(x)

        # 4. 取最後一個時間步進行預測
        final_hidden = lstm_out[:, -1]  # [batch, hidden_size]

        # 5. 生成預測
        predictions = self.predict(final_hidden)  # [batch, pred_len]

        return predictions.unsqueeze(-1)  # [batch, pred_len, 1]

