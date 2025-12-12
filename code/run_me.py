import json
import os

import matplotlib.pyplot as plt

# Import all configurations
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

        current_cfg = base_cfg.copy()
        current_cfg[param_name] = val

        # Inject model_type so it persists for Part 2
        current_cfg["model_type"] = sweep_config["model_type"]

        if hasattr(dataset, "block_size"):
            target_block_size = current_cfg.get("block_size", 64)
            dataset.block_size = target_block_size

        model = model_cls(vocab_size=vocab[3], **current_cfg)

        res = train_model(model, dataset, current_cfg)
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


def plot_flops_vs_ll(all_sweep_results, filename, title="Performance vs Compute"):
    """Plots Training FLOPs vs Validation Log-Likelihood"""
    plt.figure()

    if "flops" in all_sweep_results:
        # Case: Single Result (Deliverable 2)
        flops = [all_sweep_results["flops"]]
        ll = [-all_sweep_results["best_val_loss"]]
        plt.scatter(flops, ll, s=100, c="red", label="Word Level Model")
    else:
        # Case: Dictionary of Sweeps (Deliverable 1)
        for sweep_name, results in all_sweep_results.items():
            flops = [r["flops"] for r in results]
            ll = [-r["best_val_loss"] for r in results]
            plt.plot(flops, ll, marker="o", label=sweep_name)

    plt.xlabel("Training FLOPs")
    plt.ylabel("Validation Log-Likelihood")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_loss_for_run(run_data, title, filename):
    """Helper to plot training dynamics for a specific run"""
    plt.figure()
    plt.plot(run_data["train_loss_hist"], label="Train Loss")
    plt.plot(run_data["val_loss_hist"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(SYSTEM_CONFIG["work_dir"], filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot: {save_path}")


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # PART 1: TINY SHAKESPEARE
    # ---------------------------------------------------------
    print("\n=== DELIVERABLE 1: TINY SHAKESPEARE ===")

    ts_dir = os.path.join(SYSTEM_CONFIG["data_dir"], "tiny_shakespeare")
    ts_data = TextDataset(ts_dir, level="char", block_size=64)

    # Run all sweeps
    lin_res = run_sweep(ts_data, LINEAR_CONTEXT_SWEEP)
    mlp_width_res = run_sweep(ts_data, MLP_WIDTH_SWEEP)
    mlp_depth_res = run_sweep(ts_data, MLP_DEPTH_SWEEP)
    attn_res = run_sweep(ts_data, ATTENTION_SWEEP)
    tfm_heads_res = run_sweep(ts_data, TRANSFORMER_HEADS_SWEEP)
    tfm_depth_res = run_sweep(ts_data, TRANSFORMER_DEPTH_SWEEP)
    opt_res = run_sweep(ts_data, OPTIMIZER_SWEEP)
    lr_res = run_sweep(ts_data, LR_SWEEP)

    print(f"\n--- GENERATING DELIVERABLE 1 ARTIFACTS ---")

    # --- [D1 0.2] LOSS PLOTS FOR EACH ARCHITECTURE ---
    # We find the best run within each family to plot

    # 1. Linear
    best_lin = min(lin_res, key=lambda x: x["best_val_loss"])
    plot_loss_for_run(best_lin, "Linear Model Training Loss", "d1_linear_loss.png")

    # 2. MLP (Combine width + depth runs)
    best_mlp = min(mlp_width_res + mlp_depth_res, key=lambda x: x["best_val_loss"])
    plot_loss_for_run(best_mlp, "MLP Training Loss", "d1_mlp_loss.png")

    # 3. Attention Only
    best_attn = min(attn_res, key=lambda x: x["best_val_loss"])
    plot_loss_for_run(
        best_attn, "Attention-Only Training Loss", "d1_attention_loss.png"
    )

    # 4. Transformer (Combine heads + depth + opt runs)
    # We include opt/lr runs here as they are also transformers
    all_tfm_runs = tfm_heads_res + tfm_depth_res + opt_res + lr_res
    best_tfm = min(all_tfm_runs, key=lambda x: x["best_val_loss"])
    plot_loss_for_run(best_tfm, "Transformer Training Loss", "d1_transformer_loss.png")

    # --- [D1 0.1] GLOBAL BEST HYPERPARAMETERS ---
    all_runs = lin_res + mlp_width_res + mlp_depth_res + attn_res + all_tfm_runs
    global_best_run = min(all_runs, key=lambda x: x["best_val_loss"])
    best_config = global_best_run["cfg"]

    settings_path = os.path.join(SYSTEM_CONFIG["work_dir"], "d1_0_1_best_settings.txt")
    with open(settings_path, "w") as f:
        f.write("Best Hyperparameter Settings Found:\n")
        f.write(json.dumps(best_config, indent=4))
        f.write(f"\n\nBest Validation Loss: {global_best_run['best_val_loss']:.4f}")

    # Save config for D2
    json_path = os.path.join(SYSTEM_CONFIG["work_dir"], "best_config_deliverable1.json")
    with open(json_path, "w") as f:
        json.dump(best_config, f, indent=4)

    # --- [D1 0.3] LL vs Hyperparameters ---
    plot_sweep_results(
        lin_res, "block_size", "Linear: Context vs LL", "d1_0_3_linear_context.png"
    )
    plot_sweep_results(
        mlp_width_res, "hidden_size", "MLP: Width vs LL", "d1_0_3_mlp_width.png"
    )
    plot_sweep_results(
        tfm_heads_res, "n_head", "Transformer: Heads vs LL", "d1_0_3_tfm_heads.png"
    )

    # --- [D1 0.4] LL vs FLOPs ---
    all_res_flops = {
        "Linear": lin_res,
        "MLP (Width)": mlp_width_res,
        "Attention Only": attn_res,
        "Transformer (Heads)": tfm_heads_res,
    }
    plot_flops_vs_ll(all_res_flops, "d1_0_4_flops_vs_ll.png")

    # --- [D1 0.5] Hamlet Generation ---
    hamlet_prompt = "HAMLET:"
    hamlet_text = generate_text(global_best_run["model"], ts_data, hamlet_prompt, 200)
    save_text_sample(
        "d1_0_5_hamlet_generation.txt", hamlet_prompt, hamlet_text, "Best Model"
    )

    # ---------------------------------------------------------
    # PART 2: WORD LEVEL (PTB & WIKITEXT)
    # ---------------------------------------------------------
    print("\n=== DELIVERABLE 2: WORD LEVEL (PTB & WIKITEXT) ===")

    with open(json_path, "r") as f:
        d2_config = json.load(f)

    d2_config["epochs"] = max(d2_config.get("epochs", 5), 10)
    print(f"Training Config for D2: {d2_config}")

    datasets_to_run = [
        ("PTB", "ptb", "the school announced that", "d2_ptb"),
        ("WikiText", "wikitext-2", "The history of machine learning begins", "d2_wiki"),
    ]

    for name, folder_name, prompt, file_id in datasets_to_run:
        path = os.path.join(SYSTEM_CONFIG["data_dir"], folder_name)
        if not os.path.exists(path):
            print(f"Skipping {name} (Directory missing at {path})")
            continue

        print(f"\nTraining on {name}...")

        dataset = TextDataset(path, level="word", block_size=d2_config["block_size"])
        vocab = dataset.get_vocab()

        model_cls = get_model_class(d2_config["model_type"])
        model = model_cls(vocab_size=vocab[3], **d2_config)

        res = train_model(model, dataset, d2_config)

        single_result = {
            "best_val_loss": min(res["val_loss"]),
            "flops": res["total_flops"],
        }

        # [D2 0.1] Training Loss vs Epochs
        plt.figure()
        plt.plot(res["train_loss"], label="Train Loss")
        plt.plot(res["val_loss"], label="Val Loss")
        plt.title(f"{name} Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SYSTEM_CONFIG["work_dir"], f"{file_id}_0_1_loss.png"))
        plt.close()

        # [D2 0.2] LL vs FLOPs (Single Point Plot)
        plot_flops_vs_ll(
            single_result,
            f"{file_id}_0_2_flops_vs_ll.png",
            title=f"{name} Perf vs Compute",
        )

        # [D2 0.3 / 0.4] Text Generation
        print(f"--- {name} Generated Sample ---")
        gen_text = generate_text(model, dataset, prompt, 100)
        print(gen_text)

        if name == "PTB":
            out_file = "d2_0_3_ptb_generation.txt"
        else:
            out_file = "d2_0_4_wiki_generation.txt"

        save_text_sample(
            out_file, prompt, gen_text, f"{d2_config['model_type']} ({name})"
        )

    print("\nDone! All files for Deliverable 1 & 2 are in 'report_src/' folder.")
