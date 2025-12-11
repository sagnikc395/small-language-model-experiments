# code/trainer.py
import torch
import time
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))

    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 32)

    # Trackers
    train_losses = []
    total_flops = 0
    flops_per_step = model.estimate_flops() * batch_size

    model.train()

    # We will fix iterations per epoch for consistency across small datasets
    steps_per_epoch = 100

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
            optimizer.step()

            epoch_loss += loss.item()
            total_flops += flops_per_step
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / steps_per_epoch
        train_losses.append(avg_train_loss)

    # Final Evaluation
    val_loss = estimate_loss(model, dataset, batch_size)
    return {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_loss,
        "total_flops": total_flops,
        "config": config,
    }


@torch.no_grad()
def estimate_loss(model, dataset, batch_size=32, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = dataset.get_batch("val", batch_size)
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
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
