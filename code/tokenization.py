from torch.utils.data import DataLoader, Dataset
import torch


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])  # type: ignore


class CharDataset(Dataset):
    def __init__(self, text, tokenizer, context_len):
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + 1 + self.context_len]
        return x, y
