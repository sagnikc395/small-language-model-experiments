import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device):
    m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    m = m.masked_fill(m == 1, float("-inf"))
    return m.unsqueeze(0).unsqueeze(0)


## Linear Predictor Model
## A single linear (softmax regression) layer


class LinearModel(nn.Module):
    def __init__(self, vocab, ctx):
        super().__init__()
        self.vocab = vocab
        self.ctx = ctx
        self.fc = nn.Linear(ctx * vocab, ctx * vocab)

    def forward(self, x):
        onehot = F.one_hot(x, self.vocab).float()
        flat = onehot.view(x.size(0), -1)
        out = self.fc(flat)
        return out.view(x.size(0), self.ctx, self.vocab)


## Multi Layer Perceptron
## At least 3 layers , with nonlinear activations


class MLP(nn.Module):
    def __init__(self, vocab, ctx, hdim=256, nlayers=2):
        super().__init__()
        layers = []
        din = ctx * vocab
        for _ in range(nlayers):
            layers.append(nn.Linear(din, hdim))
            layers.append(nn.ReLU())
            din = hdim

        layers.append(nn.Linear(din, ctx * vocab))
        self.net = nn.Sequential(*layers)
        self.vocab = vocab
        self.ctx = ctx

    def forward(self, x):
        onehot = F.one_hot(x, self.vocab).float()
        flat = onehot.view(x.size(0), -1)
        out = self.net(flat)
        return out.view(x.size(0), self.ctx, self.vocab)


## Single Self Attention
## Basic building block of the more layered approach to use multi-head self-attention model


class SelfAttentionLM(nn.Module):
    def __init__(self, vocab, ctx, embed=128, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, embed)
        self.pos = nn.Embedding(ctx, embed)
        self.qkv = nn.Linear(embed, embed * 3)
        self.proj = nn.Linear(embed, embed)
        self.fc = nn.Linear(embed, vocab)
        self.h = heads
        self.ctx = ctx

    def forward(self, x):
        B, T = x.shape
        tok = self.tok(x)
        pos = self.pos(torch.arange(T, device=x.device))[None, :, :]
        h = tok + pos

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = q.size(-1) // self.h
        q = q.view(B, T, self.h, hd).transpose(1, 2)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att += causal_mask(T, x.device)
        att = att.softmax(-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.proj(out)

        return self.fc(out)


## Transformer Block
## we use this to build the transformer language model
class TransformerBlock(nn.Module):
    def __init__(self, embed, heads, mlp_h):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed)
        self.ln2 = nn.LayerNorm(embed)
        self.att = SelfAttentionLM
        self.mlp = nn.Sequential(
            nn.Linear(embed, mlp_h), nn.ReLU(), nn.Linear(mlp_h, embed)
        )

    def forward(self, x, mask):
        x = x + self.att(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


### Transformer Language Model
class TransformerLM(nn.Module):
    def __init__(self, vocab, ctx, layers=2, embed=128, heads=4, mlp_h=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, embed)
        self.pos = nn.Embedding(ctx, embed)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed, heads, mlp_h) for _ in range(layers)]
        )

        self.ln = nn.LayerNorm(embed)
        self.fc = nn.Linear(embed, vocab)
        self.ctx = ctx

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None, :, :]
        mask = causal_mask(T, x.device)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.fc(self.ln(h))
