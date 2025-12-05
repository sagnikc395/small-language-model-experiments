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


def compute_training_flops(model, train_loader, context_len, epochs):
    n_params = sum(p.numel() for p in model.parameters())
    total_tokens = len(train_loader.dataset) * context_len
    flops_per_epoch = 2 * n_params * total_tokens
    return flops_per_epoch * epochs

#our character level works on characters, for word-level , seperate helper
def generate_words(model,tokenizer,prompt,context_len,max_new_words):
    model.eval()

    #encode the prompt to ids
    prompt_ids = tokenizer.encode(prompt)
    x = torch.tensor(prompt_ids,device=DEVICE).unsqueeze(0) # (1,T0)

    for _ in range(max_new_words):
        x_cond = x[:,-context_len:] # crop to the context_len
        logits = model(x_cond)
        next_token_logits = logits[:,-1,:]
        next_token = torch.argmax(next_token_logits,dim=-1)

        x = torch.cat([x,next_token.unsqueeze(0)],dim=1)
    
    ids = x[0].tolist()
    return tokenizer.decode(ids)