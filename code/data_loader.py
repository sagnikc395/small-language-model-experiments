# code/data_loader.py
import torch
import os


class TextDataset:
    def __init__(self, dir_path, level="char", block_size=32):
        self.block_size = block_size
        self.level = level
        self.dir_path = dir_path

        # Define paths based on your structure
        train_path = os.path.join(dir_path, "train.txt")
        val_path = os.path.join(dir_path, "valid.txt")

        # 1. Load Training Data
        if not os.path.exists(train_path):
            print(f"ERROR: Train file not found at {train_path}")
            # Dummy fallback to prevent crash
            train_text = "dummy data " * 1000
        else:
            with open(train_path, "r", encoding="utf-8") as f:
                train_text = f.read()

        # 2. Tokenization & Vocab Building (From Train only)
        if level == "char":
            train_tokens = list(train_text)
        else:  # word level
            train_tokens = train_text.split()

        self.chars = sorted(list(set(train_tokens)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Loaded {dir_path}")
        print(f" - Vocab size: {self.vocab_size}")

        # 3. Encode Training Data
        self.train_data = self._encode(train_tokens)
        print(f" - Train tokens: {len(self.train_data)}")

        # 4. Load & Encode Validation Data
        if os.path.exists(val_path):
            with open(val_path, "r", encoding="utf-8") as f:
                val_text = f.read()
            if level == "char":
                val_tokens = list(val_text)
            else:
                val_tokens = val_text.split()
            self.val_data = self._encode(val_tokens)
            print(f" - Val tokens:   {len(self.val_data)}")
        else:
            print(
                f"WARNING: No valid.txt found in {dir_path}. Using train set as validation."
            )
            self.val_data = self.train_data

    def _encode(self, tokens):
        # Helper to encode tokens to integers, skipping unknowns
        return torch.tensor(
            [self.stoi[t] for t in tokens if t in self.stoi], dtype=torch.long
        )

    def get_vocab(self):
        return self.chars, self.stoi, self.itos, self.vocab_size

    def get_batch(self, split, batch_size=32):
        # Select appropriate tensor
        data = self.train_data if split == "train" else self.val_data

        # Generate random chunks
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y
