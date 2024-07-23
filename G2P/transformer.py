import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.d_model)

        out = self.fc(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.layernorm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(q=x, k=enc_output, v=enc_output, mask=src_mask)
        x = self.layernorm2(x + self.dropout(cross_attn_output))

        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, tokenizer=None, config=None, stress=False):
        super(TransformerBlock, self).__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.input_vocab_size = tokenizer.get_vocab_size()
        self.target_vocab_size = tokenizer.get_vocab_size()
        self.d_model = config.get('D_MODEL', 512)
        self.num_heads = config.get('NUM_HEADS', 8)
        self.num_encoder_layers = config.get('NUM', 6)
        self.num_decoder_layers = config.get('NUM', 6)
        self.d_ff = config.get('D_FF', 2048)
        self.dropout = config.get('DROPOUT', 0.1)
        self.stress = stress

        self.encoder_embedding = nn.Embedding(self.input_vocab_size, self.d_model)
        self.decoder_embedding = nn.Embedding(self.target_vocab_size, self.d_model)

        self.pos_embedding = PositionalEncoding(self.d_model, config.get('MAX_LEN', 32))

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in
             range(self.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in
             range(self.num_decoder_layers)])

        self.fc_out = nn.Linear(self.d_model, self.target_vocab_size)

    def encode(self, src, src_mask):
        src = self.pos_embedding(self.encoder_embedding(src))
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.pos_embedding(self.decoder_embedding(tgt))
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return tgt

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        output = self.fc_out(output)
        return output
