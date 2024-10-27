import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len  # 輸入的序列長度
        self.pred_len = configs.pred_len  # 預測的序列長度
        self.enc_in = configs.enc_in  # 輸入特徵數
        self.enc_out = configs.enc_out # 輸出
        self.d_model = configs.d_model  # 隱藏層維度
        self.dropout = configs.dropout  # dropout 機率

        # Segmentation 設定
        self.seg_len = configs.seg_len  # 每個 segment 的長度
        self.seg_num_x = self.seq_len // self.seg_len  # 輸入的 segment 數量
        self.seg_num_y = self.pred_len  # 預測的時間步長

        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_out, self.d_model // 2))

        # 修改這裡的輸出層，使其輸出 1 個降雨機率
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        seq_last = x[:, -1:, :].detach()
        x = x.permute(0, 2, 1)  # b,c,s

        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # 生成位置和通道嵌入
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_out, 1, 1),  # out feature, 1, 1
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)   # bcm, 1, d

        # RNN 處理
        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y // (self.enc_in // self.enc_out)).view(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 使用 self.predict 預測，並將形狀調整為 (batch_size, pred_len, 1)
        y = self.predict(hy).view(batch_size, self.enc_out, self.pred_len)  # 調整輸出形狀為 (b, pred_len, 1)

        y = y.permute(0, 2, 1)

        return y
