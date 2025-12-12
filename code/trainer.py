import torch
from tqdm import tqdm


# Detect Device (including Mac MPS)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = get_device()
print(f"Using device: {device}")


def train_model(model, dataset, config):
    model = model.to(device)

    # --- 1. OPTIMIZER SELECTION ---
    lr = config.get("lr", 1e-3)
    opt_name = config.get("optimizer", "adamw").lower()
    weight_decay = config.get("weight_decay", 1e-2)  # Standard default for AdamW

    if opt_name == "sgd":
        # SGD usually requires momentum to work on these tasks
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        # Default to AdamW (Best for Transformers)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 32)

    # Trackers
    train_losses = []
    val_losses = []
    total_flops = 0

    # estimate FLOPS per step
    flops_per_step = model.estimate_flops() * batch_size

    model.train()

    # lr schduler
    steps_per_epoch = 100
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(
        f"--> Starting: {config.get('name', 'Model')} | Opt: {opt_name.upper()} | LR: {lr} | Device: {device}"
    )

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(
            range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )

        for _ in pbar:
            xb, yb = dataset.get_batch("train", batch_size=batch_size)
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # optimizer.step()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total_flops += flops_per_step

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = epoch_loss / steps_per_epoch
        train_losses.append(avg_train_loss)

        # checking the validataion loss every epoch
        current_val_loss = estimate_loss(model, dataset, batch_size)
        val_losses.append(current_val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f} | Val Loss {current_val_loss:.4f}"
        )
    # Final Evaluation
    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "total_flops": total_flops,
        "config": config,
    }


@torch.no_grad()
def estimate_loss(model, dataset, batch_size=32, eval_iters=50):
    """
    Estimates the loss on the validation set using `eval_iters` batches.
    """
    model.eval()  # Switch to evaluation mode (disable dropout)
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = dataset.get_batch("val", batch_size)
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()  # Switch back to training mode
    return losses.mean().item()


def generate_text(model, dataset, prompt, max_tokens=100):
    model.eval()
    model.to(device)

    if dataset.level == "char":
        ids = [dataset.stoi.get(c, 0) for c in prompt]
    else:
        ids = [dataset.stoi.get(w, 0) for w in prompt.split()]

    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    gen_idx = model.generate(idx, max_new_tokens=max_tokens)
    gen_list = gen_idx[0].tolist()

    if dataset.level == "char":
        return "".join([dataset.itos.get(i, "") for i in gen_list])
    else:
        return " ".join([dataset.itos.get(i, "") for i in gen_list])
