# code/run_me.py
import os
import matplotlib.pyplot as plt
import torch

from config import (
    SYSTEM_CONFIG,
    LINEAR_SWEEP,
    MLP_SWEEP,
    TRANSFORMER_SWEEP,
    D2_SETTINGS,
)
from data_loader import TextDataset
from models import LinearModel, MLPModel, TransformerModel
from trainer import train_model, generate_text

# Ensure results directory exists
os.makedirs(SYSTEM_CONFIG["work_dir"], exist_ok=True)


def get_model_class(type_str):
    if type_str == "linear":
        return LinearModel
    if type_str == "mlp":
        return MLPModel
    if type_str == "transformer":
        return TransformerModel
    raise ValueError(f"Unknown model type: {type_str}")


def run_sweep(dataset, sweep_config):
    print(f"\n>>> Running Sweep: {sweep_config['name']}")

    param_name = sweep_config["sweep_param"]
    values = sweep_config["sweep_values"]
    base_cfg = sweep_config["base"]
    model_cls = get_model_class(sweep_config["model_type"])

    results = []

    vocab = dataset.get_vocab()

    for val in values:
        print(f"   Testing {param_name} = {val}")

        # Update config with the sweep value
        current_cfg = base_cfg.copy()
        current_cfg[param_name] = val

        if hasattr(dataset, "block_size"):
            target_block_size = current_cfg.get("block_size", 64)
            dataset.block_size = target_block_size
            print(f"    -> Dataset block_size updated to {dataset.block_size}")

        # Init model
        model = model_cls(vocab_size=vocab[3], **current_cfg)

        # Train
        res = train_model(model, dataset, current_cfg)

        # Store essential data
        results.append(
            {
                "val": val,
                "val_loss": res["val_loss"],
                "flops": res["total_flops"],
                "train_loss_hist": res["train_loss"],
                "model": res["model"],
                "cfg": current_cfg,
            }
        )

    return results


def plot_sweep_results(results, param_name, title, filename):
    x = [r["val"] for r in results]
    y = [-r["val_loss"] for r in results]  # Log Likelihood

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("Validation Log-Likelihood")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], filename))
    plt.close()


def plot_flops_vs_ll(all_sweep_results, filename):
    plt.figure()
    for sweep_name, results in all_sweep_results.items():
        flops = [r["flops"] for r in results]
        ll = [-r["val_loss"] for r in results]
        plt.plot(flops, ll, marker="o", label=sweep_name)

    plt.xlabel("Training FLOPs")
    plt.ylabel("Validation Log-Likelihood")
    plt.title("Performance vs Compute")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], filename))
    plt.close()


def plot_loss_curves(results, title, filename):
    plt.figure()
    for r in results:
        label = f"Val: {r['val']}"
        plt.plot(r["train_loss_hist"], label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], filename))
    plt.close()


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # --- PART 1: Tiny Shakespeare ---
    print("=== DELIVERABLE 1: TINY SHAKESPEARE ===")

    # Pointing to the DIRECTORY, not the file
    ts_dir = os.path.join(SYSTEM_CONFIG["data_dir"], "tiny_shakespeare")

    # This will load train.txt and valid.txt inside that directory
    ts_data = TextDataset(ts_dir, level="char", block_size=64)

    # Run Sweeps
    lin_res = run_sweep(ts_data, LINEAR_SWEEP)
    mlp_res = run_sweep(ts_data, MLP_SWEEP)
    tfm_res = run_sweep(ts_data, TRANSFORMER_SWEEP)

    # 0.2: Plot Training Loss vs Epochs (Using Transformer sweep)
    plot_loss_curves(tfm_res, "Transformer Training Loss", "d1_transformer_loss.png")

    # 0.3: Plot Log-Likelihood vs Hyperparameters
    plot_sweep_results(
        lin_res, "block_size", "Linear: Context Length vs LL", "d1_linear_sweep.png"
    )
    plot_sweep_results(
        mlp_res, "hidden_size", "MLP: Hidden Dim vs LL", "d1_mlp_sweep.png"
    )
    plot_sweep_results(
        tfm_res, "n_head", "Transformer: Heads vs LL", "d1_transformer_sweep.png"
    )

    # 0.4: Plot Log-Likelihood vs FLOPs
    all_res = {"Linear": lin_res, "MLP": mlp_res, "Transformer": tfm_res}
    plot_flops_vs_ll(all_res, "d1_flops_vs_ll.png")

    # 0.5: Hamlet Generation (Using best Transformer)
    best_tfm = min(tfm_res, key=lambda x: x["val_loss"])
    print("\n--- HAMLET GENERATION (Best Transformer) ---")
    print(generate_text(best_tfm["model"], ts_data, "HAMLET:", 200))

    # --- PART 2: PTB & WikiText ---
    print("\n=== DELIVERABLE 2: WORD LEVEL ===")

    best_config = best_tfm["cfg"]
    print(f"Selected Best Architecture Config: {best_config}")

    # Only need Folder Name and Prompt now
    datasets = [
        ("PTB", "ptb", "the school announced that"),
        ("WikiText", "wikitext-2", "The history of machine learning begins"),
    ]

    d2_results = {}

    for name, folder_name, prompt in datasets:
        path = os.path.join(SYSTEM_CONFIG["data_dir"], folder_name)
        if not os.path.exists(path):
            print(f"Skipping {name} (Directory missing at {path})")
            continue

        print(f"\nTraining on {name}...")

        # Load Word-Level Data
        dataset = TextDataset(path, level="word", block_size=best_config["block_size"])
        vocab = dataset.get_vocab()

        # Instantiate Model with Best Config + Word Vocab
        model = TransformerModel(vocab_size=vocab[3], **best_config)

        # Train
        res = train_model(model, dataset, D2_SETTINGS)
        d2_results[name] = res

        # 0.1 Plot Loss
        plt.figure()
        plt.plot(res["train_loss"])
        plt.title(f"{name} Training Loss")
        plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], f"d2_{name}_loss.png"))
        plt.close()

        # 0.3 / 0.4 Generation
        print(f"--- {name} Generated Sample ---")
        print(generate_text(model, dataset, prompt, 100))
        print("---------------------------------")

    print("\nDone! Check 'results/' folder for plots.")
