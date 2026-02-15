import torch
import torch.nn as nn
import math


# ==================================================
# Multi-Head Attention
# ==================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = d_model

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

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.n_heads * self.d_k)

        output = self.fc(context)

        return output


# ==================================================
# Position-wise Feed Forward
# ==================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# ==================================================
# Encoder Layer
# ==================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, d_k)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ==================================================
# BERT Model
# ==================================================
class BERT(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        d_k,
        n_segments,
        vocab_size,
        max_len,
        device
    ):
        super().__init__()

        self.device = device
        self.d_model = d_model

        # Embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)

        # Encoder layers
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, d_k)
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(0.1)

    # --------------------------------------------------
    # Create Attention Mask
    # --------------------------------------------------
    def get_attention_mask(self, input_ids):
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        return mask

    # --------------------------------------------------
    # Forward Pass
    # --------------------------------------------------
    def forward(self, input_ids, segment_ids):
        batch_size, seq_len = input_ids.size()

        pos_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)

        embedding = (
            self.tok_embed(input_ids)
            + self.pos_embed(pos_ids)
            + self.seg_embed(segment_ids)
        )

        x = self.dropout(embedding)

        mask = self.get_attention_mask(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        return x  # [batch_size, seq_len, d_model]

    # --------------------------------------------------
    # Token-level Output (for SBERT)
    # --------------------------------------------------
    def get_last_hidden_state(self, input_ids, segment_ids):
        return self.forward(input_ids, segment_ids)

    # --------------------------------------------------
    # Sentence Embedding (Mean Pooling)
    # --------------------------------------------------
    def get_sentence_embedding(self, input_ids, segment_ids):
        outputs = self.forward(input_ids, segment_ids)
        return torch.mean(outputs, dim=1)
