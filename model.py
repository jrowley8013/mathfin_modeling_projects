"""Portfolio optimization models."""

import copy
import math

import torch
from torch import nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab: int, kernel_size: int, padding: int):
        super(Embeddings, self).__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=vocab,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding_mode="circular",
            padding=padding,
            bias=False,
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        x = x.float()
        output_tensor = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return output_tensor * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Trainable normalization layer."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask=None,
    dropout=None,
    silent=True,
):
    d_k = K.size(-1)
    QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if not silent:
        print("\t\tQK:", QK.size())
    if mask is not None:
        QK = QK.masked_fill(mask == 0, -1e9)
    QK_softmax = QK.softmax(dim=-1)
    if not silent:
        print("\t\tQK_softmax:", QK_softmax.size())
    if dropout is not None:
        QK_softmax = nn.Dropout(dropout)(QK_softmax)
    QK_softmax_V = torch.matmul(QK_softmax, V)
    if not silent:
        print("\t\tQK_softmax_V", QK_softmax_V.size())
    return QK_softmax_V, QK_softmax


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, D_q, D_k, D_v, d_out, h, dropout=0.1, silent=True):
        super(MultiHeadedAttention, self).__init__()

        self.linear_Q = nn.Linear(d_in, D_q, bias=False)
        self.linear_K = nn.Linear(d_in, D_k, bias=False)
        self.linear_V = nn.Linear(d_in, D_v, bias=False)
        self.linear_Wo = nn.Linear(D_v, d_out)

        self.d_in = d_in
        self.D_q = D_q
        self.D_k = D_k
        self.D_v = D_v
        self.d_out = d_out
        self.h = h
        self.dropout = dropout
        self.silent = silent

    def forward(self, query, key, value, mask=None):
        if not self.silent:
            print("MultiHeadedAttention")
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        B, T_q, _ = query.shape
        B, T_k, _ = key.shape
        B, T_v, _ = value.shape
        if not self.silent:
            print("\tBefore Linear Mapping:")
            print("\t\tquery (= X):", query.size())
            print("\t\tkey (= X):", key.size())
            print("\t\tvalue (= X):", value.size())
        Q = self.linear_Q(query).view(B, T_q, self.h, -1).transpose(1, 2)
        K = self.linear_K(key).view(B, T_k, self.h, -1).transpose(1, 2)
        V = self.linear_V(value).view(B, T_v, self.h, -1).transpose(1, 2)
        if not self.silent:
            print("\t\tWq:", self.linear_Q.weight.size())
            print("\t\tWk:", self.linear_K.weight.size())
            print("\t\tWv:", self.linear_V.weight.size())
            print("\tAfter Linear Mapping:")
            print("\t\tQ (= X * Wq)", Q.size())
            print("\t\tK (= X * Wk)", K.size())
            print("\t\tV (= X * Wv):", V.size())

            print("\n\tCalculating Attention: start")
        Z_i, _ = attention(Q, K, V, mask=mask, dropout=self.dropout, silent=self.silent)
        if not self.silent:
            print("\tCalculation Attention: end\n")
            print("\tZ_i:", Z_i.size())
        Z_i_concat = Z_i.transpose(1, 2).contiguous().view(nbatches, -1, self.D_v)
        if not self.silent:
            print("\tZ_i_concat:", Z_i_concat.size())

        Z = self.linear_Wo(Z_i_concat)

        del query
        del key
        del value
        if not self.silent:
            print("\tWo:", self.linear_Wo.weight.size())
            print("\tZ (= Z_i_concat * Wo):", Z.size())
        return Z


class SublayerConnection(nn.Module):
    """Encoder/Decoder sublayer connection layer."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        sublayer1 = lambda x: self.self_attn(x, x, x, mask)
        sublayer2 = self.feed_forward
        x = self.sublayer_connections[0](x, sublayer1)
        return self.sublayer_connections[1](x, sublayer2)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        sublayer1 = lambda x: self.self_attn(x, x, x, tgt_mask)
        sublayer2 = lambda x: self.src_attn(x, memory, memory, src_mask)
        sublayer3 = self.feed_forward
        x = self.sublayer_connections[0](x, sublayer1)
        x = self.sublayer_connections[1](x, sublayer2)
        return self.sublayer_connections[2](x, sublayer3)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        embedded_src = self.src_embed(src)
        return self.encoder(embedded_src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        embedded_tgt = self.tgt_embed(tgt)
        return self.decoder(embedded_tgt, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


class GRN(nn.Module):
    def __init__(self, size):
        super(GRN, self).__init__()
        self.elu = nn.ELU()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)
        self.layernorm = LayerNorm(size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        g2 = self.elu(self.linear1(z))  # g2 = ELU(W2*z + b2)
        g1 = self.linear2(g2)  # g1 = W1 * g2 + b1
        glu_output = g1 * self.sigmoid(g1)  # GLU (g1)
        grn_output = self.layernorm(z + glu_output)
        return grn_output


def abs_softmax(x):
    """Returns weights with absolute values that sum to 1"""
    x_exp = torch.exp(x.abs())
    x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
    return torch.sign(x) * x_exp / x_exp_sum


class ConditionalFlatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # For a 3D input (e.g., [20, 16, 10]), flatten from dimension 1
            # so the output becomes [20, 16*10] => [20, 160].
            return x.flatten(start_dim=1)
        elif x.dim() == 2:
            # For a 2D input (e.g., [20, 16]), flatten from dimension 0
            # so the output becomes [20*16] => [320].
            return x.flatten(start_dim=0)
        else:
            raise ValueError(f"Expected input tensor to be 2D or 3D, but got shape {x.shape}")


class Generator(nn.Module):
    def __init__(self, d_flatten, vocab):
        super(Generator, self).__init__()
        # Replacing nn.Flatten(start_dim=1) with our custom conditional flattening module.
        self.flatten = ConditionalFlatten()
        self.proj = nn.Linear(d_flatten, vocab)

    def forward(self, x):
        # Flatten the input using ConditionalFlatten,
        # then project it and apply the abs_softmax function.
        proj_x = self.proj(self.flatten(x))
        return abs_softmax(proj_x)


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    kernel_size: int,
    padding: int,
    N: int,
    d_model: int,
    h: int,
    output_seq_length: int,
    dropout: float = 0.1,
):
    dc = copy.deepcopy
    attn = MultiHeadedAttention(
        d_in=d_model,
        D_q=d_model * h,
        D_k=d_model * h,
        D_v=d_model * h,
        d_out=d_model,
        h=h,
        dropout=dropout,
    )
    ff = GRN(size=d_model)
    position = PositionalEncoding(d_model=d_model, dropout=dropout)

    encoder = Encoder(
        EncoderLayer(
            size=d_model, self_attn=dc(attn), feed_forward=dc(ff), dropout=dropout
        ),
        N=N,
    )
    decoder = Decoder(
        DecoderLayer(
            size=d_model,
            self_attn=dc(attn),
            src_attn=dc(attn),
            feed_forward=dc(ff),
            dropout=dropout,
        ),
        N=N,
    )
    src_embed = nn.Sequential(
        Embeddings(
            d_model=d_model, vocab=src_vocab, kernel_size=kernel_size, padding=padding
        ),
        dc(position),
    )
    tgt_embed = nn.Sequential(
        Embeddings(
            d_model=d_model, vocab=tgt_vocab, kernel_size=kernel_size, padding=padding
        ),
        dc(position),
    )
    generator = Generator(d_flatten=(output_seq_length * d_model), vocab=tgt_vocab)
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=generator,
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class LSTMModel(nn.Module):
    """LSTM Model."""

    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    generator: Generator
    embed_layer: nn.Sequential
    lstm_layer: nn.LSTM

    def __init__(
        self, input_size, hidden_size, num_layers, dropout, generator, embed_layer
    ):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.generator = generator

        self.embed_layer = embed_layer

        self.lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.lstm_layer(x)
        return x


def make_lstm_model(
    d_model: int,
    n_assets: int,
    kernel_size: int,
    padding: int,
    output_seq_length: int,
    hidden_size: int,
    num_lstm_layers: int,
    dropout: float = 0.1,
) -> nn.Module:
    """Make a simple LSTM model."""
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    dc = copy.deepcopy

    embed = nn.Sequential(
        Embeddings(
            d_model=d_model, vocab=n_assets, kernel_size=kernel_size, padding=padding
        ),
        dc(position),
    )

    generator = Generator(d_flatten=(output_seq_length * hidden_size), vocab=n_assets)

    model = LSTMModel(
        input_size=d_model,
        hidden_size=hidden_size,
        num_layers=num_lstm_layers,
        dropout=dropout,
        generator=generator,
        embed_layer=embed,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y):
        x = self.generator(x)
        # loss is defined as the Sharpe Ratio of the batch
        sloss = self.criterion(x, y)
        return sloss.data, sloss
