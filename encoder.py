import torch.nn as nn
from attention import MultiHeadAttentionLayer
from config import device, NHEAD, EMBED_DIM

class EncoderBlock(nn.Module):
    def __init__(self, emb_size, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionLayer(emb_size, nhead, dropout=dropout, device=device)
        self.linear1 = nn.Linear(emb_size, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src