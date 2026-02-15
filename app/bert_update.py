import torch
import torch.nn as nn
import math


# --------------------------------------------------
# Multi-Head Self Attention
# --------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k

        self.W_Q = nn.Linear(d_model, n_heads * d_k)
        self.W_K = nn.Linear(d_model, n_heads * d_k)
        self.W_V = nn.Linear(d_model, n_heads * d_k)

        self.fc = nn.Linear(n_heads * d_k, d_model)

    def forward(self, Q, K, V, mask=None):

        batch_size = Q.size(0)

        q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        output = self.fc(context)

        return output


# --------------------------------------------------
# Feed Forward Network (IMPORTANT: fc1 / fc2 names)
# --------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# --------------------------------------------------
# Encoder Layer
# --------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


# --------------------------------------------------
# BERT Model (IMPORTANT: tok_embed / pos_embed / seg_embed)
# --------------------------------------------------
class BERT(nn.Module):
    def __init__(self,
                 n_layers,
                 n_heads,
                 d_model,
                 d_ff,
                 d_k,
                 n_segments,
                 vocab_size,
                 max_len,
                 device):

        super().__init__()

        self.device = device

        # ⚠ Must match saved model names
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, d_k)
            for _ in range(n_layers)
        ])

        #self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, segment_ids):

        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand_as(input_ids)

        x = (
            self.tok_embed(input_ids) +
            self.pos_embed(positions) +
            self.seg_embed(segment_ids)
        )

        for layer in self.layers:
            x = layer(x)

        #x = self.norm(x)

        return x

    def get_last_hidden_state(self, input_ids, segment_ids):
        return self.forward(input_ids, segment_ids)
