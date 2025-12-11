import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sized, Tuple, Union, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from architectures import MLP, LinearRegressionModel, SelfAttentionLM, TransformerLM

# --- Config Imports ---
from config import (
    EXPERIMENT_CONFIG_DELIVERABLE_1,
    EXPERIMENT_CONFIG_DELIVERABLE_2,
    ArchitectureType,
    OptimizerType,
)
from evaluation import (
    DEVICE,
    compute_loss,
    compute_training_flops,
    generate,
    generate_words,
)
from tokenization import CharDataset, CharTokenizer, WordDataset, WordTokenizer

# --- Local Imports ---
from utils import (
    REPORT_DIR,
    SHAKESPEARE_DATA_PATH,
    load_shakespeare_data,
    load_word_level_data,
    logger,
)


def create_model(
    arch: Union[str, ArchitectureType], vocab_size: int, ctx: int, conf: Dict[str, Any]
) -> Tuple[nn.Module, Any, str]:
    """
    Instantiates a model and determines what parameter is being swept for plotting.
    Returns: (Model, Value_to_Plot, Label_for_Plot)
    """
    param_val = conf.get("varying_param", None)
    param_label = conf.get("param_label", "Hyperparameter")
    model_type = conf.get("model_type", arch)

    match model_type:
        case "linear":
            act = conf.get("activation", "identity")
            model = LinearRegressionModel(vocab_size, ctx, activation=act)
            if param_val is None:
                param_val = ctx
                param_label = "Context Length"
            return model, param_val, param_label

        case "mlp":
            hidden_dim = conf["hidden_dims"][0]
            model = MLP(vocab_size, ctx, conf["hidden_dims"])
            if param_val is None:
                param_val = hidden_dim
                param_label = "Hidden Dimension"
            return model, param_val, param_label

        case "self_attention":
            model = SelfAttentionLM(
                vocab_size, ctx, conf["embed_dim"], conf["num_heads"]
            )
            if param_val is None:
                param_val = conf["num_heads"]
                param_label = "Num Heads"
            return model, param_val, param_label

        case "transformer":
            model = TransformerLM(
                vocab_size,
                ctx,
                conf["embed_dim"],
                conf["num_heads"],
                conf["mlp_hidden"],
                conf["num_layers"],
            )
            if param_val is None:
                param_val = conf["num_layers"]
                param_label = "Num Layers"
            return model, param_val, param_label

        case "optimization":
            model = TransformerLM(
                vocab_size,
                ctx,
                embed_dim=128,
                num_heads=4,
                mlp_hidden=256,
                num_layers=2,
            )
            return model, conf.get("optimizer", "adam"), "Optimizer Type"

        case _:
            raise ValueError(f"Unknown architecture/model_type: {model_type}")


def get_optimizer(
    model: nn.Module, optimizer_type: OptimizerType, lr: float
) -> torch.optim.Optimizer:
    """Factory to create optimizer strategies."""
    opt_lower = optimizer_type.lower()
    match opt_lower:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr)
        case "sgd-momentum":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        case "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        case _:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def save_plots(
    exp_name: str,
    results: List[Dict[str, Any]],
    best_curve: List[float],
    report_dir: str,
):
    """Generates and saves the 3 required plots for Deliverable 1."""

    # 1. Training Loss Curve
    plt.figure()
    plt.plot(best_curve)
    plt.title(f"{exp_name} Best Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(report_dir, f"d1_{exp_name}_loss.png"))
    plt.close()

    # Data prep
    x_vals = [r["param"] for r in results]
    y_ll = [r["ll"] for r in results]
    x_flops = [r["flops"] for r in results]
    param_name = results[0]["param_name"]

    # 2. Hyperparameter vs Test LL
    plt.figure()
    if isinstance(x_vals[0], str):
        plt.bar(x_vals, y_ll, color="skyblue")
    else:
        plt.plot(x_vals, y_ll, marker="o", linestyle="-", color="orange")

    plt.title(f"{exp_name}: Test LL vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Test LL")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(report_dir, f"d1_{exp_name}_hyperparam.png"))
    plt.close()

    # 3. FLOPs vs Test LL
    plt.figure()
    plt.scatter(x_flops, y_ll, s=100, color="green")
    plt.xscale("log")
    plt.title(f"{exp_name}: LL vs FLOPs")
    plt.xlabel("Training FLOPs (Log Scale)")
    plt.ylabel("Test LL")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(report_dir, f"d1_{exp_name}_flops.png"))
    plt.close()


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 5,
    lr: float = 1e-3,
    context_len: Optional[int] = None,
    optimizer_name: OptimizerType = "adam",
) -> Dict[str, List[float]]:
    """Universal training loop."""
    model = model.to(DEVICE)
    optimizer = get_optimizer(model, optimizer_name, lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_ll": [], "flops": []}

    flops_per_epoch = 0.0
    if context_len is not None:
        n_params = sum(p.numel() for p in model.parameters())
        dataset = cast(Sized, train_loader.dataset)
        total_tokens = len(dataset) * context_len
        flops_per_epoch = 2 * n_params * total_tokens

    logger.info(f"Training {model.__class__.__name__} [{optimizer_name}] on {DEVICE}")
    cumulative_flops = 0.0

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
        history["train_loss"].append(epoch_loss)

        log_msg = f"Epoch {ep + 1}/{epochs} | Train Loss: {epoch_loss:.4f}"

        if test_loader and context_len:
            cumulative_flops += flops_per_epoch
            test_ll = compute_loss(model, test_loader, criterion)
            history["test_ll"].append(test_ll)
            history["flops"].append(cumulative_flops)
            log_msg += f" | Test LL: {test_ll:.4f}"

        print(log_msg)

    return history


def run_deliverable_1() -> Dict[str, Any]:
    print("\n======= STARTING DELIVERABLE 1: Tiny Shakespeare Sweeps =====")

    train_text, valid_text, test_text = load_shakespeare_data(SHAKESPEARE_DATA_PATH)
    tokenizer = CharTokenizer(train_text)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "d1_best_generations.txt")
    json_path = os.path.join(REPORT_DIR, "deliverable_1_best_config.json")

    # --- GLOBAL BEST TRACKER ---
    global_best_ll = float("inf")
    global_best_arch = "transformer"
    global_best_config = {}

    with open(report_path, "w") as f:
        for exp_name, configs in EXPERIMENT_CONFIG_DELIVERABLE_1.items():
            print(f"\n>>> Running Experiment Group: {exp_name.upper()}")

            results_data = []
            group_best_ll = float("inf")
            group_best_artifacts = {}

            for conf in configs:
                print(f"    Running Config: {conf['name']}...")
                ctx = conf["context_len"]
                epochs = 3

                train_ds = CharDataset(train_text, tokenizer, ctx)
                test_ds = CharDataset(test_text, tokenizer, ctx)
                train_loader = torch.utils.data.DataLoader(
                    train_ds, batch_size=32, shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)

                model, param_val, param_name = create_model(
                    exp_name, tokenizer.vocab_size, ctx, conf
                )

                opt_name = conf.get("optimizer", "adam")
                history = train_model(
                    model,
                    train_loader,
                    epochs=epochs,
                    lr=conf["lr"],
                    optimizer_name=opt_name,
                )

                test_ll = compute_loss(model, test_loader, nn.CrossEntropyLoss())
                flops = compute_training_flops(model, train_loader, ctx, epochs)

                results_data.append(
                    {
                        "param": param_val,
                        "param_name": param_name,
                        "ll": test_ll,
                        "flops": flops,
                    }
                )

                if test_ll < group_best_ll:
                    group_best_ll = test_ll
                    # Use correct arguments: (model, tokenizer, context, context_len, length)
                    gen_text = generate(model, tokenizer, "HAMLET: ", ctx, length=100)
                    group_best_artifacts = {
                        "config": conf,
                        "curve": history["train_loss"],
                        "gen": gen_text,
                    }

                # --- TRACK GLOBAL BEST ---
                if test_ll < global_best_ll:
                    print(
                        f"    [New Global Best!] {exp_name} ({test_ll:.4f}) beat previous ({global_best_ll:.4f})"
                    )
                    global_best_ll = test_ll
                    global_best_arch = conf.get("model_type", exp_name)
                    global_best_config = conf
                # -------------------------

            save_plots(
                exp_name, results_data, group_best_artifacts["curve"], REPORT_DIR
            )

            f.write(f"=== {exp_name.upper()} ===\n")
            f.write(f"Best Config: {group_best_artifacts['config']}\n")
            f.write(f"{group_best_artifacts['gen']}\n\n")

    # --- SAVE BEST CONFIG TO JSON ---
    winner_info = {
        "arch": global_best_arch,
        "config": global_best_config,
        "test_ll": global_best_ll,
    }

    with open(json_path, "w") as jf:
        json.dump(winner_info, jf, indent=4)

    print(f"\n[INFO] Best config saved to: {json_path}")
    print(f"Deliverable 1 Complete. Global Winner: {global_best_arch}")

    return winner_info


def run_deliverable_2():
    print("\n======= STARTING DELIVERABLE 2: Word Level Experiments =====")

    # 1. Attempt to load Best Config from JSON
    json_path = os.path.join(REPORT_DIR, "deliverable_1_best_config.json")

    arch_to_use = "transformer"  # Default
    config_to_use = {
        "embed_dim": 256,
        "num_heads": 4,
        "mlp_hidden": 512,
        "num_layers": 4,
    }

    if os.path.exists(json_path):
        print(f"[INFO] Found D1 Best Config at {json_path}. Loading...")
        with open(json_path, "r") as f:
            data = json.load(f)
            arch_to_use = data.get("arch", "transformer")
            config_to_use = data.get("config", config_to_use)
            print(f"      -> Using Architecture: {arch_to_use}")
            print(f"      -> Using Hyperparams: {config_to_use}")
    else:
        print("[WARN] No D1 config found. Using default robust Transformer settings.")

    context_len = 64
    epochs = 5
    batch_size = 32

    for task in EXPERIMENT_CONFIG_DELIVERABLE_2:
        print(f"\n--- Running Task: {task['name']} ---")

        train_text, valid_text, test_text = load_word_level_data(task["path"])
        tokenizer = WordTokenizer(train_text, vocab_size=task["vocab"])

        train_ds = WordDataset(train_text, tokenizer, context_len)
        test_ds = WordDataset(test_text, tokenizer, context_len)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

        # Dynamic Model Creation
        model, _, _ = create_model(
            arch_to_use, tokenizer.vocab_size, context_len, config_to_use
        )

        # Train
        history = train_model(
            model,
            train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=5e-4,
            context_len=context_len,
        )

        # Plot Loss
        plt.figure()
        plt.plot(range(1, epochs + 1), history["train_loss"], marker="o")
        plt.title(f"{task['name']} ({arch_to_use}) Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(REPORT_DIR, f"d2_{task['name']}_loss.png"))
        plt.close()

        # Plot LL vs FLOPs Trajectory
        plt.figure()
        plt.plot(history["flops"], history["test_ll"], marker="o", linestyle="-")
        plt.title(f"{task['name']} ({arch_to_use}) Test LL vs Training FLOPs")
        plt.xlabel("Cumulative FLOPs")
        plt.ylabel("Test LL")
        plt.xscale("log")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(REPORT_DIR, f"d2_{task['name']}_ll_flops.png"))
        plt.close()

        # Generation
        gen_text = f"=== {task['name']} GENERATIONS ===\n"
        print(f"Generating samples for {task['name']}...")
        for prompt in task["prompts"]:
            sample = generate_words(model, tokenizer, prompt, context_len, 100)
            gen_text += f"\nPrompt: '{prompt}'\n{sample}\n"
            print(f" > {prompt}...")

        with open(os.path.join(REPORT_DIR, f"d2_{task['name']}_gen.txt"), "w") as f:
            f.write(gen_text)

    print(f"Deliverable 2 Complete. Results in {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1", action="store_true", help="Run Deliverable 1")
    parser.add_argument("--d2", action="store_true", help="Run Deliverable 2")
    args = parser.parse_args()

    # Step 1: Run D1 if needed
    if args.d1 or (not args.d1 and not args.d2):
        run_deliverable_1()

    # Step 2: Run D2 if needed (it will read the JSON created by D1)
    if args.d2 or (not args.d1 and not args.d2):
        run_deliverable_2()
