import torch

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def compute_loss(model, loader, criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)


def generate(model, tokenizer, context, context_len, length=100):
    model.eval()
    ids = tokenizer.encode(context)
    x = torch.tensor(ids, device=DEVICE).unsqueeze(0)

    for _ in range(length):
        logits = model(x[:, -context_len:])
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(x[0].tolist())
