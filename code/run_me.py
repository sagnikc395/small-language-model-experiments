import json
import os

import matplotlib.pyplot as plt

# Import configurations
from config import (
    ATTENTION_SWEEP,
    LINEAR_CONTEXT_SWEEP,
    LR_SWEEP,
    MLP_DEPTH_SWEEP,
    MLP_WIDTH_SWEEP,
    OPTIMIZER_SWEEP,
    SYSTEM_CONFIG,
    TRANSFORMER_DEPTH_SWEEP,
    TRANSFORMER_HEADS_SWEEP,
)
from data_loader import TextDataset
from models import AttentionModel, LinearModel, MLPModel, TransformerModel
from trainer import generate_text, train_model

# Ensure results directory exists
os.makedirs(SYSTEM_CONFIG["work_dir"], exist_ok=True)


def get_model_class(type_str):
    if type_str == "linear":
        return LinearModel
    if type_str == "mlp":
        return MLPModel
    if type_str == "attention":
        return AttentionModel
    if type_str == "transformer":
        return TransformerModel
    raise ValueError(f"Unknown model type: {type_str}")


def save_text_sample(filename, prompt, text, model_name):
    """Saves generated text to a file in the work directory."""
    filepath = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write("-" * 40 + "\n")
        f.write(text + "\n")
        f.write("-" * 40 + "\n")
    print(f"Saved generated text to: {filepath}")


def run_sweep(dataset, sweep_config):
    """
    Runs a hyperparameter sweep and returns results.
    """
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

        current_cfg["model_type"] = sweep_config["model_type"]

        # Handle block_size updates (requires updating dataset property)
        if hasattr(dataset, "block_size"):
            target_block_size = current_cfg.get("block_size", 64)
            dataset.block_size = target_block_size

        # Init model
        model = model_cls(vocab_size=vocab[3], **current_cfg)

        # Train
        res = train_model(model, dataset, current_cfg)

        # Extract Metrics
        best_val_loss = min(res["val_loss"])

        results.append(
            {
                "val": val,
                "best_val_loss": best_val_loss,
                "val_loss_hist": res["val_loss"],
                "train_loss_hist": res["train_loss"],
                "flops": res["total_flops"],
                "model": res["model"],
                "cfg": current_cfg,
            }
        )

    return results


def plot_sweep_results(results, param_name, title, filename):
    x = [r["val"] for r in results]
    y = [-r["best_val_loss"] for r in results]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(param_name)
    plt.ylabel("Validation Log-Likelihood")
    plt.title(title)
    plt.grid(True)
    save_path = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_flops_vs_ll(all_sweep_results, filename):
    plt.figure()
    for sweep_name, results in all_sweep_results.items():
        flops = [r["flops"] for r in results]
        ll = [-r["best_val_loss"] for r in results]
        plt.plot(flops, ll, marker="o", label=sweep_name)

    plt.xlabel("Training FLOPs")
    plt.ylabel("Validation Log-Likelihood")
    plt.title("Performance vs Compute")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_loss_curves(results, title, filename):
    plt.figure()
    for r in results:
        label = f"Val: {r['val']}"
        plt.plot(r["train_loss_hist"], label=label)

    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    # --- PART 1: Tiny Shakespeare ---
    print("\n=== DELIVERABLE 1: TINY SHAKESPEARE ===")

    ts_dir = os.path.join(SYSTEM_CONFIG["data_dir"], "tiny_shakespeare")
    ts_data = TextDataset(ts_dir, level="char", block_size=64)

    # 1. Linear Context Sweep
    lin_res = run_sweep(ts_data, LINEAR_CONTEXT_SWEEP)
    plot_sweep_results(
        lin_res, "block_size", "Linear: Context Length vs LL", "d1_linear_context.png"
    )

    # 2. MLP Sweeps
    mlp_width_res = run_sweep(ts_data, MLP_WIDTH_SWEEP)
    plot_sweep_results(
        mlp_width_res, "hidden_size", "MLP: Width vs LL", "d1_mlp_width.png"
    )

    mlp_depth_res = run_sweep(ts_data, MLP_DEPTH_SWEEP)
    plot_sweep_results(
        mlp_depth_res, "n_layers", "MLP: Depth vs LL", "d1_mlp_depth.png"
    )

    # 3. Attention-Only Sweep
    attn_res = run_sweep(ts_data, ATTENTION_SWEEP)
    plot_sweep_results(
        attn_res, "n_head", "Attention-Only: Heads vs LL", "d1_attn_heads.png"
    )

    # 4. Transformer Sweeps
    tfm_heads_res = run_sweep(ts_data, TRANSFORMER_HEADS_SWEEP)
    plot_sweep_results(
        tfm_heads_res, "n_head", "Transformer: Heads vs LL", "d1_tfm_heads.png"
    )

    # Plot Training Loss vs Epochs
    plot_loss_curves(
        tfm_heads_res,
        "Transformer Training Loss (Heads Sweep)",
        "d1_tfm_loss_curves.png",
    )

    tfm_depth_res = run_sweep(ts_data, TRANSFORMER_DEPTH_SWEEP)
    plot_sweep_results(
        tfm_depth_res, "n_layers", "Transformer: Depth vs LL", "d1_tfm_depth.png"
    )

    # 5. Optimization Sweeps
    opt_res = run_sweep(ts_data, OPTIMIZER_SWEEP)
    plot_sweep_results(
        opt_res, "optimizer", "Optimizer Choice vs LL", "d1_optimizer.png"
    )

    lr_res = run_sweep(ts_data, LR_SWEEP)
    plt.figure()
    plt.semilogx(
        [r["val"] for r in lr_res], [-r["best_val_loss"] for r in lr_res], marker="o"
    )
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Validation Log-Likelihood")
    plt.title("Learning Rate vs LL")
    plt.grid(True)
    plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], "d1_lr_sweep.png"))
    plt.close()

    # Plot FLOPs vs LL
    all_res_flops = {
        "Linear": lin_res,
        "MLP (Width)": mlp_width_res,
        "Attention Only": attn_res,
        "Transformer (Heads)": tfm_heads_res,
    }
    plot_flops_vs_ll(all_res_flops, "d1_flops_vs_ll.png")

    # --- Identify Best Model from Deliverable 1 ---
    all_runs = (
        lin_res
        + mlp_width_res
        + mlp_depth_res
        + attn_res
        + tfm_heads_res
        + tfm_depth_res
        + opt_res
        + lr_res
    )
    best_run = min(all_runs, key=lambda x: x["best_val_loss"])
    best_config = best_run["cfg"]

    print(
        f"\n--- BEST MODEL FOUND: {best_config['model_type']} (Val Loss: {best_run['best_val_loss']:.4f}) ---"
    )
    print(f"Config: {best_config}")

    # Save Best Config to JSON for Deliverable 2
    json_path = os.path.join(SYSTEM_CONFIG["work_dir"], "best_config_deliverable1.json")
    with open(json_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"Saved best configuration to {json_path}")

    # Generate Hamlet Sample
    hamlet_prompt = "HAMLET:"
    hamlet_text = generate_text(best_run["model"], ts_data, hamlet_prompt, 200)
    save_text_sample(
        "d1_hamlet_generation.txt",
        hamlet_prompt,
        hamlet_text,
        "Best Model (TinyShakespeare)",
    )

    # --- PART 2: PTB & WikiText ---
    print("\n=== DELIVERABLE 2: WORD LEVEL (PTB & WIKITEXT) ===")

    # 1. Load the Best Config from File
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            d2_config = json.load(f)
        print("Loaded architecture from best_config_deliverable1.json")
    else:
        print("Warning: Config file not found, falling back to default D2 settings.")
        # Fallback if something went wrong
        d2_config = {
            "model_type": "transformer",
            "n_embd": 128,
            "n_head": 4,
            "n_layers": 2,
            "lr": 5e-4,
            "optimizer": "adamw",
            "block_size": 64,
        }

    # 2. Adjust for Word-Level Training
    # Word level datasets are harder; we might want to ensure sufficient epochs even if D1 was short.
    d2_config["epochs"] = max(d2_config.get("epochs", 5), 10)
    print(f"Training Config for D2: {d2_config}")

    datasets_to_run = [
        ("PTB", "ptb", "the school announced that"),
        ("WikiText", "wikitext-2", "The history of machine learning begins"),
    ]

    for name, folder_name, prompt in datasets_to_run:
        path = os.path.join(SYSTEM_CONFIG["data_dir"], folder_name)
        if not os.path.exists(path):
            print(f"Skipping {name} (Directory missing at {path})")
            continue

        print(f"\nTraining on {name}...")

        # Load Word-Level Data
        dataset = TextDataset(path, level="word", block_size=d2_config["block_size"])
        vocab = dataset.get_vocab()

        # Instantiate Model with Best Config
        model_cls = get_model_class(d2_config["model_type"])
        model = model_cls(vocab_size=vocab[3], **d2_config)

        # Train
        res = train_model(model, dataset, d2_config)

        # 0.1 Plot Loss
        plt.figure()
        plt.plot(res["train_loss"], label="Train")
        plt.plot(res["val_loss"], label="Val")
        plt.title(f"{name} Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], f"d2_{name}_loss.png"))
        plt.close()

        # 0.3 / 0.4 Generation & SAVE TO FILE
        print(f"--- {name} Generated Sample ---")
        gen_text = generate_text(model, dataset, prompt, 100)
        print(gen_text)
        print("---------------------------------")

        save_filename = f"d2_{name.lower().replace(' ', '_')}_generation.txt"
        save_text_sample(
            save_filename, prompt, gen_text, f"{d2_config['model_type']} ({name})"
        )

    print("\nDone! Check 'report_src/' folder for plots and text files.")
