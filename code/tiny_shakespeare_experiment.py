import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    def __init__(self, vocab_size, context_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.linear = nn.Linear(context_len * vocab_size, context_len * vocab_size)

    def forward(self, x):
        B,T  = x.shape
        assert T == self.context_len

        # convert the tokens to one hot encoding 
        onehot = F.one_hot(x,num_classes=self.vocab_size).float()
        flat = onehot.view(B,-1)
        logits = self.linear(flat).view(B,T,self.vocab_size)
        return logits 


class MLP(nn.Module):
    def __init__(self,vocab_size,context_len,hidden_dims=[512,512,512]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len

        layers = []
        input_dim = context_len * vocab_size

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim,h))
            layers.append(nn.ReLU())
            input_dim = h 

        layers.append(nn.Linear(input_dim,context_len * vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        B,T = x.shape 
        onehot = F.one_hot(x,num_classes=self.vocab_size).float()
        flat = onehot.view(B,-1)
        out = self.net(flat)
        return out.view(B,T,self.vocab_size)
    



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dum must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # linear layers to get Q,K,V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape

        # linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape to multi-head format
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores,dim=-1)
        out= attn @ V 

        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.out_proj(out)
    
class SelfAttentionLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim=128, num_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_len, embed_dim)

        self.attn = MultiHeadSelfAttention(embed_dim, num_heads,dropout=0)
        self.ln = nn.LayerNorm(embed_dim)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)                        # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=x.device))[None, :, :]
        h = tok + pos                                   # add positional enc

        h = self.attn(h)
        h = self.ln(h)

        logits = self.fc(h)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim,num_heads,mlp_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim,num_heads,dropout=0)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden,embed_dim)
        )

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x 
    
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_len,embed_dim=128,num_heads=4,mlp_hidden=256,num_layers=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size,embed_dim)
        self.pos_emb = nn.Embedding(context_len,embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim,num_heads,mlp_hidden)
        for _ in range(num_layers)] )
        
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim,vocab_size)

    def forward(self,x):
        B,T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T,device=x.device))[None,:,:]
        h = tok + pos 

        for layer in self.layers:
            h = layer(h)

        h = self.ln(h)
        logits = self.fc(h)
        return logits 
    
    