from torch.utils.data import Dataset
import torch
from collections import Counter


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


# adding word tokenizer and word dataset for deliverable 2
# for word level
class WordTokenizer:
    def __init__(self, text, vocab_size, min_freq=1):
        # simple whitespace tokenization; replace with out PTB/WT2 tokenization
        tokens = text.strip().split()

        counter = Counter(tokens)

        # special tokens
        self.pad_token = "<pad>"  # for padding
        self.unk_token = "<unk>"  # for unknown

        vocab = [self.pad_token, self.unk_token]

        # filter the vocab by frequency or vocab_sizes
        if vocab_size is not None:
            vocab += [w for w, _ in counter.most_common(vocab_size - 2)]
        else:
            vocab += [w for w, c in counter.items() if c >= min_freq]

        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [
            self.stoi.get(tok, self.stoi[self.unk_token])
            for tok in text.strip().split()
        ]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids if i in self.itos)


class WordDataset(Dataset):
    def __init__(self, text, tokenizer, context_len):
        self.context_len = context_len
        self.tokenizer = tokenizer

        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + 1 + self.context_len]
        return x, y
