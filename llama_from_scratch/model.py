from ctypes import c_void_p

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass 
from typing import Optional 

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number heads for queries
    n_kv_heads: Optional[int] = None # number heads for k and v
    vocab_size: int = -1 
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None # feed forward layer dim
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, 'Uneven head dimension!'

    # build theta parameter: theta_i = 10000 ^ (-2*(i-1)/dim) for i = [1, 2, ..., dim /2]
    # shape (head_dim /2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # shape (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct positions ('m' parameter) of the token inside the sentence
    # shape (seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using the outer product
    # shape: (seq_len) outer product* (head_dim /2) -> (seq_len, head_dim /2)
    freqs = torch.outer(m, theta).float()
    # compute complex numbers in the polar form:
    # c = R * exp(i * m * theta)
    # where R = 1 as follows:
    # shape (seq_len, head_dim / 2) -> (seq_len, head_dim /2)
    freqs_complex: torch.Tensor = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # take 2 consecutive dimensions, group them
    # then transform into complex number
    # (b, seq_len, H, head_dim) -> (b, seq_len, H, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (b, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (b, seq_len, H, head_dim /2)
    x_rotated = x_complex * freqs_complex
    # (b, seq_len, H, head_dim / 2) -> (b, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # flatten
    # (b, seq_len, H, head_dim / 2, 2) -> (b, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # (bs, seq_len, n_kv_heads, 1, head_dim)
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # shape (b, seq_len, dim) * (b, seq_len, 1) = (b, seq_len, dim)
        # rsqrt : 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # dim * (b, seq_len, dim) = (b, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Number of heads for the Keys and Values
        self.n_kv_heads: int = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Number of heads for the queries
        self.n_heads_q: int = args.n_heads
        # Number of times the Keys and Values are repeated to match the Queries
        self.n_rep: int = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        args.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # bs, 1, dim
        batch_size, seq_len, _  = x.shape

        # (bs, 1, dim) -> (bs, 1, h_q * head_dim)
        xq = self.wq(x)
        # # (bs, 1, dim) -> (bs, 1, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (bs, 1, h_q * head_dim) -> (bs, 1, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (bs, 1, h_kv * head_dim) -> (bs, 1, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # No impact on tensor's shape
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xv = apply_rotary_embeddings(xv, freqs_complex, device=x.device)

        # Replace cached Query entry for the token
        next_pos = start_pos + seq_len
        self.cache_k[:batch_size, start_pos: next_pos] = xk
        self.cache_v[:batch_size, start_pos: next_pos] = xv

        # Retrieve all cached Keys and Values
        # (bs, seq_len_kv, h_kv, head_dim)
        keys = self.cache_k[:batch_size, 0: next_pos]
        values = self.cache_v[:batch_size, 0: next_pos]

        # Repeat the heads of the K and V to reach the number of heads in the Queries
        # in Grouped Query Attention, there are more Q matrices than K and V
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (bs, 1, h_q, head_dim) -> (bs, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = xk.transpse(1, 2)
        values = xv.transpose(1, 2)

        # (bs, h_q, 1, head_dim) @ (bs, h_q, head_dim, seq_len_kv) --> (bs, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (bs, h_q, 1, seq_len) @ (bs, h_q, seq_len_kv, head_dim) --> (bs, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (bs, h_q, 1, head_dim) -> (bs, 1, h_q, head_dim) -> (bs, 1, head_dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))

        return self.wo(output) # b, 1, dim -> b, 1, dim


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round hidden_dim to the nearest multiple of the 'multiple_of' argument
        # e.g. hidden_dim = 7; multiple_of = 5; [(7 + 5 - 1) // 5] * 5 = [2] * 5 = 10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalize BEFORE self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # normalize BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (b, seq_len, dim) + (b, seq_len, dim) --> (b, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))

        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, 'Vocab size must be set'

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # b, seq_len
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, 'Only one token at a time can be processed.'

        # b, seq_len -> b, seq_len, dim
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()