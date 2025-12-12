import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Base Class ---
class BaseModel(nn.Module):
    """Base class with generation capability"""

    def __init__(self, block_size) -> None:
        super().__init__()
        # Store block_size so generate() can access it
        self.block_size = block_size

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            # idx shape is (B, T)
            idx_cond = idx[:, -self.block_size :]

            # Get predictions
            logits, _ = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- 1. Linear Model ---
class LinearModel(BaseModel):
    def __init__(self, vocab_size, block_size, n_embd, **kwargs):
        # FIX: Pass block_size to parent
        super().__init__(block_size)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # Flatten input (B, T, C) -> (B, T*C)
        self.head = nn.Linear(block_size * n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # SAFETY CHECK: LinearModel requires exact block_size length
        # because the Linear layer dimensions are fixed.
        if T > self.block_size:
            idx = idx[:, -self.block_size :]
        elif T < self.block_size:
            raise ValueError(
                f"Input sequence length {T} is too short for LinearModel (needs {self.block_size})"
            )

        # Re-calculate shapes after cropping
        B, T = idx.shape

        x = self.token_embedding(idx).view(B, -1)  # Flattens to (B, T*C)
        logits = self.head(x).unsqueeze(1)  # (B, 1, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets[:, -1].view(-1)
            )
        return logits, loss

    def estimate_flops(self):
        return (
            2
            * (self.block_size * self.token_embedding.embedding_dim)
            * self.head.out_features
        )


# --- 2. MLP Model ---
class MLPModel(BaseModel):
    def __init__(self, vocab_size, block_size, n_embd, hidden_size, n_layers, **kwargs):
        super().__init__(block_size)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        layers = []
        input_dim = block_size * n_embd
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_size, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # --- FIX: Crop Context ---
        if T > self.block_size:
            idx = idx[:, -self.block_size :]
        elif T < self.block_size:
            raise ValueError(
                f"Input sequence length {T} is too short for MLPModel (needs {self.block_size})"
            )

        x = self.token_embedding(idx).view(B, -1)
        logits = self.net(x).unsqueeze(1)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets[:, -1].view(-1)
            )
        return logits, loss

    def estimate_flops(self):
        flops = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                flops += 2 * layer.in_features * layer.out_features
        return flops


# --- 3. Transformer Model ---
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register buffer to ensure it is part of state_dict but not a parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Scaled Dot-Product Attention
        wei = q @ k.transpose(-2, -1) * (C**-0.5)

        # Masking: Ensure we don't look ahead.
        # Slicing [:T, :T] handles cases where input length T < block_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # type: ignore

        wei = F.softmax(wei, dim=-1)

        # add dropout
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, n_embd)

        # added dropout for the output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.ReLU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """
    A Transformer block STRIPPED of the FeedForward layer.
    It only performs Self-Attention.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ln1 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Only Attention + Residual + Norm
        # No FeedForward layer here!
        x = x + self.sa(self.ln1(x))
        return x


class AttentionModel(BaseModel):
    def __init__(
        self, vocab_size, block_size, n_embd, n_head, n_layers, dropout=0.2, **kwargs
    ):
        super().__init__(block_size)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Use the specific AttentionBlock (No FF)
        self.blocks = nn.Sequential(
            *[
                AttentionBlock(n_embd, n_head, block_size, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    def estimate_flops(self):
        return sum(p.numel() for p in self.parameters()) * 2


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, n_embd, block_size, dropout=dropout
        )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Norm formulation (Karpathy/GPT-2 style)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(BaseModel):
    def __init__(
        self, vocab_size, block_size, n_embd, n_head, n_layers, dropout=0.2, **kwargs
    ):
        # FIX: Pass block_size to parent
        super().__init__(block_size)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # dropout for the embeddings
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, n_head, block_size, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        # Ensure position embeddings match current sequence length T
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # embedding dropout
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

    def estimate_flops(self):
        # Rough proxy: Num Params * 2
        return sum(p.numel() for p in self.parameters()) * 2
