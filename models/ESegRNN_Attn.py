import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 基本參數
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.enc_out = configs.enc_out
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        # Segmentation參數
        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len

        # 特徵嵌入
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # 修正多層GRU的input_size設定
        self.gru1 = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 第二層GRU的input_size要是第一層的hidden_size*2
        self.gru2 = nn.GRU(
            input_size=self.d_model * 2,  # 因為第一層是雙向的,所以輸入維度是d_model*2
            hidden_size=self.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 降維層,用於處理GRU的輸出
        self.dim_reduction = nn.Linear(self.d_model * 2, self.d_model)

        # 自注意力機制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=4,
            dropout=self.dropout
        )

        # 位置和通道編碼
        self.pos_emb = nn.Parameter(torch.randn(self.seq_len, self.d_model))

        # 預測層
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model * 1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
            nn.Flatten(start_dim=1),    # 從 b, p, 1 變成 b, p
            nn.Linear(self.seq_len, self.pred_len),    # 從 b, p 變成 b, o
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 1. 特徵預處理
        x = x.permute(0, 2, 1)  # (batch, channel, seq_len)

        neko = x.reshape(batch_size, self.seq_len, -1)

        # 2. 分段和嵌入
        x = self.valueEmbedding(x.reshape(batch_size, self.seq_len, -1))

        # 3. 第一層GRU
        gru1_output, _ = self.gru1(x)  # 輸出維度: (batch, seq_len, d_model*2)
        gru1_output = torch.relu(gru1_output)

        # 4. 第二層GRU
        gru2_output, _ = self.gru2(gru1_output)  # 輸出維度: (batch, seq_len, d_model*2)
        gru2_output = torch.relu(gru2_output)

        # 5. 降維
        gru_output = self.dim_reduction(gru2_output)  # 降到 (batch, seq_len, d_model)

        # 6. 自注意力處理
        gru_output = gru_output.permute(1, 0, 2)  # (seq_len, batch, d_model)
        attn_output, _ = self.self_attention(gru_output, gru_output, gru_output)
        attn_output = attn_output.permute(1, 0, 2)  # (batch, seq_len, d_model)

        # 7. 位置和通道編碼
        pos_emb = self.pos_emb.unsqueeze(0)

        pos_emb = pos_emb.repeat(batch_size, 1, 1)

        # 8. 合併特徵
        final_features = torch.cat([attn_output, pos_emb], dim=-1)

        # 9. 預測
        y = self.predict(final_features).unsqueeze(2)
        y = y.view(batch_size, self.pred_len, -1)  # (batch_size, pred_len, 1)

        return y
