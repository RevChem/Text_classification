import torch.nn as nn
from config import MAX_LEN
from positionalencoding import PositionalEncoding
from encoder import EncoderBlock

class TextTransformer(nn.Module):
    def __init__(self, num_classes, vocab_size, emb_size=128, nhead=2,
                 num_layers=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, max_len=MAX_LEN)

        self.layers = nn.ModuleList([
            EncoderBlock(emb_size, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)

        for layer in self.layers:
            src = layer(src)

        output = src.mean(dim=1)
        return self.classifier(output)