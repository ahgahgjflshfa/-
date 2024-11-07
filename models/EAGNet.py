import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 基本參數
        self.seq_len = configs.seq_len  # 80
        self.pred_len = configs.pred_len  # 4
        self.enc_in = configs.enc_in  # 4個特徵
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        # 1. 特徵嵌入層
        self.feature_embedding = nn.ModuleDict({
            'pressure': nn.Linear(1, self.d_model // 4),
            'temp': nn.Linear(1, self.d_model // 4),
            'wind': nn.Linear(1, self.d_model // 4),
            'humidity': nn.Linear(1, self.d_model // 4)
        })

        # 2. 單一卷積層（減少多層卷積）來提取局部特徵
        self.conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)

        # 3. 簡化時間注意力（只保留一層全局注意力）
        self.global_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=4,  # 減少頭數
            dropout=self.dropout
        )

        # 4. 單層GRU層（簡化GRU結構）
        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model // 2,
            num_layers=1,  # 使用單層
            batch_first=True,
            bidirectional=True
        )

        # 5. 簡化預測層
        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.pred_len),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 特徵分離和嵌入
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        features = x.split(1, dim=1)  # 分離四個特徵

        embedded_features = []
        for feat, (name, embedder) in zip(features, self.feature_embedding.items()):
            embedded = embedder(feat.permute(0, 2, 1))  # [batch, seq_len, d_model//4]
            embedded_features.append(embedded)

        # 2. 特徵融合
        x = torch.cat(embedded_features, dim=-1)  # [batch, seq_len, d_model]

        # 3. 卷積處理局部特徵
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, d_model]

        # 4. 全局時間注意力
        x = x.transpose(0, 1)  # 轉換維度以符合 MultiheadAttention 輸入格式
        x, _ = self.global_attention(x, x, x)  # [seq_len, batch, d_model]
        x = x.transpose(0, 1)  # 轉換回 [batch, seq_len, d_model]

        # 5. 單層GRU
        gru_out, _ = self.gru(x)

        # 6. 最終預測
        final_features = gru_out[:, -1]  # 取最後一個時間步的GRU輸出
        output = self.predict(final_features)  # [batch, pred_len]

        return output.unsqueeze(-1)  # [batch, pred_len, 1]

