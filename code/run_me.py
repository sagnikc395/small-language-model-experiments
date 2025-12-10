import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

from utils import (
    logger,
    load_shakespeare_data,
    load_word_level_data,
    SHAKESPEARE_DATA_PATH,
    PTB_DIR,
    WT2_DIR,
    REPORT_DIR,
)

from evaluation import (
    DEVICE,
    compute_loss,
    compute_training_flops,
    generate,
    generate_words,
)

from tokenization import CharTokenizer, CharDataset, WordTokenizer, WordDataset

from architectures import LinearRegressionModel, MLP, SelfAttentionLM, TransformerLM

from config import EXPERIMENT_CONFIG_DELIVERABLE_1


# generic training loop for all
def train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []

    logger.info(f"Training {model.__class__.__name__} on {DEVICE}")

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

        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_loss)

        # printing each epoch
        print(f"   Epoch {ep + 1}/{epochs} | Loss : {epoch_loss:.4f}")
    return train_losses


# hyperparameter sweeps for shakespeare corpus experiments
def run_deliverable_1():
    print("\n======= STARTING DELIVERABLE 1: Tiny Shakespeare Sweeps =====")

    # setup data
    train_text, valid_text, test_text = load_shakespeare_data(SHAKESPEARE_DATA_PATH)
    tokenizer = CharTokenizer(train_text)

    # store the results in a dictironary that we can use to plot at the end
    experiment_results = {}

    for arch, configs in EXPERIMENT_CONFIG_DELIVERABLE_1.items():
        print(f"\n>>> Architecture: {arch.upper()}")
        arch_data = []
        best_ll = float("inf")
        best_config = None
        best_gen = ""
        best_loss_curve = []

        for conf in configs:
            print(f"    Running {conf['name']}...")
            ctx = conf["context_len"]
            epochs = 3  # fixed this for comparision

            # load the datasets
            train_ds = CharDataset(train_text, tokenizer, ctx)
            test_ds = CharDataset(test_text, tokenizer, ctx)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=32, shuffle=True
            )
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

            if arch == "linear":
                model = LinearRegressionModel(tokenizer.vocab_size, ctx)
                param_val = ctx
                param_name = "Context Length"
            elif arch == "mlp":
                model = MLP(tokenizer.vocab_size, ctx, conf["hidden_dims"])
                param_val = conf["hidden_dims"][0]
                param_name = "Hidden Dimension"
            elif arch == "self_attention":
                model = SelfAttentionLM(
                    tokenizer.vocab_size, ctx, conf["embed_dim"], conf["num_heads"]
                )
                param_val = conf["num_heads"]
                param_name = "Num Heads"
            elif arch == "transformer":
                model = TransformerLM(
                    tokenizer.vocab_size,
                    ctx,
                    conf["embed_dim"],
                    conf["num_heads"],
                    conf["mlp_hidden"],
                    conf["num_layers"],
                )
                param_val = conf["num_layers"]
                param_name = "Num Layers"

            # train (we are passing test_loader to be consistent, though we only need final LL for D1 plots usally)
            # for  D1 sweeps efficieny, we can calcualte metrics at the very end
            history = train_model(model, train_loader, epochs=epochs, lr=conf["lr"])

            test_ll = compute_loss(model, test_loader, nn.CrossEntropyLoss)

            flops = compute_training_flops(model, train_loader, ctx, epochs)

            arch_data.append(
                {
                    "param": param_val,
                    "param_name": param_name,
                    "ll": test_ll,
                    "flops": flops,
                }
            )

            if test_ll < best_ll:
                best_ll = test_ll
                best_config = conf
                best_curve = history["train_loss"]
                best_gen = generate(model, tokenizer, "HAMLET: ", 100, ctx)

        experiment_results[arch] = {
            "data": arch_data,
            "best_curve": best_curve,
            "best_gen": best_gen,
            "best_config": best_config,
        }
        print(f"    Best {arch} Test LL: {best_ll:.4f}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "d1_best_generations.txt"), "w") as f:
        for arch, res in experiment_results.items():
            data = res["data"]

            # plotting training loss (best model)
            plt.figure()
            plt.plot(res["best_curve"])
            plt.title(f"{arch} Best Training Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(REPORT_DIR, f"f1_{arch}_loss.png"))
            plt.close()

            # plotting learning rate vs hyperparamters
            plt.figure()
            x = [d["param"] for d in data]
            y = [d["ll"] for d in data]
            plt.plot(x, y, marker="o")
            plt.title(f"{arch}: Test LL vs {data[0]['param_name']}")
            plt.xlabel(data[0]["param_name"])
            plt.ylabel("Test LL")
            plt.savefig(os.path.join(REPORT_DIR, f"d1_{arch}_hyperparam.png"))
            plt.close()

            # plotting learnin rate vs FLOPs
            plt.figure()
            x_flops = [d["flops"] for d in data]
            plt.scatter(x_flops, y, s=100)
            plt.xscale("log")
            plt.title(f"{arch}: LL vs FLOPs")
            plt.xlabel("FLOPs")
            plt.ylabel("Test LL")
            plt.savefig(os.path.join(REPORT_DIR, f"d1_{arch}_flops.png"))
            plt.close()

            f.write(
                f"=== {arch.upper()} ===\nConfig: {res['best_config']}\n{res['best_gen']}\n\n"
            )
    print(f"Deliverable 1 Complete. Results added to {REPORT_DIR}")


#
