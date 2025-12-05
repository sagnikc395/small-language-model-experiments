import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from evaluation import DEVICE, compute_loss, generate
from tokenization import CharDataset, CharTokenizer
from architectures import LinearRegressionModel
from architectures import MLP
from architectures import SelfAttentionLM
from architectures import TransformerLM


# data paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(CURRENT_DIR, "..", "report_src")
DATASETS_DIR = os.path.join(CURRENT_DIR, "..", "datasets")
SHAKESPEARE_DATA_PATH = os.path.join(DATASETS_DIR, "tiny_shakespeare")

PTB_DIR = os.path.join(DATASETS_DIR, "ptb")
WT2_DIR = os.path.join(DATASETS_DIR, "wikitext2")


# ensure datasets exist(safety!)
if not os.path.exists(SHAKESPEARE_DATA_PATH):
    raise FileNotFoundError(f"Cannot find dataset at {SHAKESPEARE_DATA_PATH}")


def load_shakespeare_data(dataset_path):
    train_file = os.path.join(dataset_path, "train.txt")
    valid_file = os.path.join(dataset_path, "valid.txt")
    test_file = os.path.join(dataset_path, "test.txt")

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    return (train_text, valid_text, test_text)


def load_word_level_data(dataset_dir):
    train_file = os.path.join(dataset_dir, "train.txt")
    valid_file = os.path.join(dataset_dir, "valid.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()

    return (train_text, valid_text, test_text)


# training loop
def train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    print(f"\n[INFO] Training Model {model.__class__.__name__}")
    for ep in range(epochs):
        model.train()
        batch_losses = []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        # average the train loss over all the batches this epoch
        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        epoch_valid_loss = compute_loss(model, valid_loader, criterion)

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

        print(
            f"Epoch {ep + 1}/{epochs} | train_loss={epoch_train_loss:.4f} | valid_loss={epoch_valid_loss:.4f}"
        )

    return train_losses, valid_losses


def run_experiment_tiny_shakespeare(
    model_type="transformer", context_len=128, batch_size=32, epochs=3
):
    print("===== {DELIVERABLE 1} =====")
    print("Training for Tiny Shakespare Experiment")
    print(f"Selected device: {DEVICE}")

    train_text, valid_text, test_text = load_shakespeare_data(SHAKESPEARE_DATA_PATH)

    # tokenizers and datasets
    tokenizer = CharTokenizer(train_text)

    # load the dataset into the character dataset first to store it
    # into embeddings fashion
    train_ds = CharDataset(train_text, tokenizer, context_len)
    valid_ds = CharDataset(valid_text, tokenizer, context_len)
    test_ds = CharDataset(test_text, tokenizer, context_len)

    # loader the data using the data loader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    match model_type:
        case "linear":
            model = LinearRegressionModel(tokenizer.vocab_size, context_len=context_len)
        case "mlp":
            model = MLP(tokenizer.vocab_size, context_len, [512, 512, 512])
        case "attention":
            model = SelfAttentionLM(
                tokenizer.vocab_size,
                context_len=context_len,
                embed_dim=128,
                num_heads=4,
            )

        case "transformer":
            model = TransformerLM(
                tokenizer.vocab_size,
                context_len=context_len,
                embed_dim=128,
                num_heads=4,
                mlp_hidden=256,
                num_layers=3,
            )

        case _:
            raise ValueError("Unknown model_type: {model_type}")

    # train
    print("\n[INFO] Started training for: {model_type}")
    train_loss, valid_loss = train_model(model, train_loader, valid_loader, epochs)

    # test the log likelihood
    test_ll = compute_loss(model, test_loader, nn.CrossEntropyLoss())
    print(f"[RESULT] Test Log Likelihood: {test_ll:.4f}")

    # generation
    sample = generate(model, tokenizer, "HAMLET: ", length=100, context_len=context_len)

    print("\n[GENERATION]")
    print(sample)

    # plot the results
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()
    plt.savefig(f"Deliverable1_{model_type}_loss_curve.png")
    plt.close()

    return model, test_ll


if __name__ == "__main__":
    run_experiment_tiny_shakespeare()
