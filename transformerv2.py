import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x, mask):
        x_normalized = self.norm1(x)
        attention_output, _ = self.attention(x_normalized, x_normalized, x_normalized, attn_mask=mask)
        x2 = x + self.dropout(attention_output)
        x_normalized2 = self.norm2(x2)
        forward_output = x2 + self.dropout(self.feed_forward(x_normalized2))
        return forward_output


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x, memory, src_mask, tgt_mask):
        x_normalized = self.norm1(x)
        attention_output, _ = self.attention(x_normalized, memory, memory, tgt_mask=tgt_mask, src_key_padding_mask=src_mask)
        x2 = x + self.dropout(attention_output)
        x_normalized2 = self.norm2(x2)
        forward_output = x2 + self.dropout(self.feed_forward(x_normalized2))
        return forward_output


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, num_layers, num_heads, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, num_heads, dropout)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, num_heads, dropout)

    def forward(self, src, trg, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg, enc_out, src_mask, tgt_mask)
        return dec_out
