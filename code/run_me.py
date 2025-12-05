import os
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


from evaluation import DEVICE, compute_loss, compute_training_flops, generate, generate_words
from tokenization import CharDataset, CharTokenizer, WordDataset, WordTokenizer
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


def run_word_level_experiment(
    dataset_name: str,
    dataset_dir: str,
    prompt: str,
    context_len: int = 64,
    batch_size: int = 32,
    epochs: int = 5,
    vocab_size: int | None = 20000,   # adjust as needed
):
    print(f"\n===== [DELIVERABLE 2] {dataset_name} word-level modeling =====")
    print(f"Device: {DEVICE}, context_len={context_len}, batch_size={batch_size}")

    # 1) Load raw text
    train_text, valid_text, test_text = load_word_level_data(dataset_dir)

    # 2) Build tokenizer on train text
    tokenizer = WordTokenizer(train_text, vocab_size=vocab_size)

    # 3) Datasets + loaders
    train_ds = WordDataset(train_text, tokenizer, context_len)
    valid_ds = WordDataset(valid_text, tokenizer, context_len)
    test_ds  = WordDataset(test_text,  tokenizer, context_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # 4) Build best model architecture from Deliverable 1
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_len=context_len,
        embed_dim=256,    # plug in your best hyperparams from Deliverable 1
        num_heads=4,
        mlp_hidden=512,
        num_layers=4,
    )

    # 5) Train
    train_losses, valid_losses = train_model(model, train_loader, valid_loader, epochs)

    # 6) Test LL
    test_ll = compute_loss(model, test_loader, nn.CrossEntropyLoss())
    print(f"[RESULT] {dataset_name} test log-likelihood: {test_ll:.4f}")

    # 7) Compute FLOPs
    flops = compute_training_flops(model, train_loader, context_len, epochs)

    # 8) Plot training loss vs epochs
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{dataset_name} – word-level Transformer")
    plt.tight_layout()
    save_path_loss = os.path.join(REPORT_DIR, f"{dataset_name}_word_train_curve.png")
    plt.savefig(save_path_loss)
    plt.close()

    # 9) LL vs FLOPs plot – for Deliverable 2, usually 1 point is boring, so
    #    you can run multiple configs and log multiple (flops, test_ll) points.
    #    Here I'll just save this single point; you can extend to a sweep.
    plt.figure()
    plt.scatter([flops], [test_ll])
    plt.xscale("log")
    plt.xlabel("Training FLOPs (log scale)")
    plt.ylabel("Test Log-Likelihood")
    plt.title(f"{dataset_name} – LL vs FLOPs")
    plt.tight_layout()
    save_path_ll_flops = os.path.join(REPORT_DIR, f"{dataset_name}_word_ll_vs_flops.png")
    plt.savefig(save_path_ll_flops)
    plt.close()

    # 10) 100-word generation with the required prompt
    gen_text = generate_words(
        model,
        tokenizer,
        prompt=prompt,
        context_len=context_len,
        max_new_words=100,
    )
    print(f"\n[GENERATION – {dataset_name}]")
    print(gen_text)

    return {
        "dataset": dataset_name,
        "test_ll": test_ll,
        "flops": flops,
        "loss_plot": save_path_loss,
        "ll_flops_plot": save_path_ll_flops,
        "generation": gen_text,
    }


if __name__ == "__main__":
    run_experiment_tiny_shakespeare()
