import torch
import torch.nn as nn


class Seq2SeqSummarizer(nn.Module):
    def __init__(self,
                 article_vocab_size,
                 summary_vocab_size,
                 embed_dim=256,
                 hidden_dim=512,
                 dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 编码器
        self.encoder_emb = nn.Embedding(article_vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            bidirectional=True,
            dropout=dropout
        )

        # 解码器（修正：删除重复定义，保留正确的输入维度）
        self.decoder_emb = nn.Embedding(summary_vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(
            embed_dim + 2 * hidden_dim,  # 输入维度正确为 256 + 1024 = 1280
            hidden_dim,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, summary_vocab_size)

        # 注意力机制和投影层保持不变
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.encoder_h_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.encoder_c_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.emb_dropout = nn.Dropout(dropout)

        # 带激活函数的双向状态投影
        self.encoder_h_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.encoder_c_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # 嵌入层dropout
        self.emb_dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        # 编码器（添加嵌入dropout）
        src_emb = self.emb_dropout(self.encoder_emb(src))
        enc_output, (h_enc, c_enc) = self.encoder_lstm(src_emb)

        # 合并双向状态（最后一层）
        h_enc = torch.cat([h_enc[-2], h_enc[-1]], dim=1)
        c_enc = torch.cat([c_enc[-2], c_enc[-1]], dim=1)
        h_dec = self.encoder_h_proj(h_enc).unsqueeze(0)
        c_dec = self.encoder_c_proj(c_enc).unsqueeze(0)

        # 解码循环
        outputs = []
        for i in range(tgt.size(0)):
            # 改进的Attention计算
            h_dec_expanded = h_dec.repeat(enc_output.size(0), 1, 1)
            energy = self.attn(
                torch.cat([enc_output, h_dec_expanded], dim=2))
            attn_weights = torch.softmax(energy, dim=0)
            context = torch.sum(enc_output * attn_weights, dim=0)

            # 解码器（添加嵌入dropout）
            tgt_emb = self.emb_dropout(self.decoder_emb(tgt[i].unsqueeze(0)))
            dec_input = torch.cat([tgt_emb, context.unsqueeze(0)], dim=2)
            dec_output, (h_dec, c_dec) = self.decoder_lstm(dec_input, (h_dec, c_dec))

            outputs.append(self.fc_out(dec_output.squeeze(0)))

        return torch.stack(outputs)