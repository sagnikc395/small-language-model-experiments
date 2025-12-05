import os 
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt 


SHAKESPEARE_DATA = "datasets/tiny_shakespeare"
S_TRAIN_FILE = os.path.join(SHAKESPEARE_DATA,"train.txt")
S_VALID_FILE = os.path.join(SHAKESPEARE_DATA,"valid.txt")
S_TEST_FILE = os.path.join(SHAKESPEARE_DATA,"test.txt")


if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# dataset and tokenizer 
class CharTokenizer:
    def __init__(self,text):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i,c in enumerate(chars)}
        self.itos = {i: c for c,i in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self,s):
        return [self.stoi[c] for c in s]
    
    def decode(self,ids):
        return "".join([self.itos[i] for i in ids]) # type: ignore
    
class CharDataset(Dataset):
    def __init__(self,text,tokenizer,context_len):
        self.data = torch.tensor(tokenizer.encode(text),dtype=torch.long)
        self.context_len = context_len
    
    def __len__(self):
        return len(self.data) - self.context_len
    
    def __getitem__(self,idx):
        x = self.data[idx: idx+ self.context_len]
        y = self.data[idx + 1: idx + 1+ self.context_len]
        return x,y 

#eval helpers 
def compute_loss(model,loader,criterion):
    model.eval()
    losses = []
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.view(01,logits.size(-1)),y.view(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)

def generate(model,tokenizer,context,length=100):
    model.eval()
    ids = tokenizer.encode(context)
    x = torch.tensor(ids,device=DEVICE).unsqueeze(0)

    context_len = len(context)
    for _ in range(length):
        logits = model(x[:,-context_len:])
        next_token = torch.argmax(logits[:,-1,:],dim=-1)
        x = torch.cat([x,next_token.unsqueeze(0)],dim=1)
    
    return tokenizer.decode(x[0].tolist())

# training loop 
def train_model(model,train_loader,valid_loader,epochs=5,lr=1e-3):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    for ep in range(epochs):
        model.train()
        batch_losses = [] 


        for x,y in train_loader:
            x ,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1,logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        #average the train loss over all the batches this epoch 
        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        epoch_valid_loss = compute_loss(model,valid_loader,criterion)

        train_losses.append(epoch_train_loss)
        valid_losses.append(epoch_valid_loss)

        print(
            f"Epoch {ep+1}/{epochs} | train_loss={epoch_train_loss:.4f} | valid_loss={epoch_valid_loss:.4f}"
        )

    return train_losses,valid_losses



def main():
    print("Training for Tiny Shakespare Experiment")
    print(f"Selected device: {DEVICE}")


if __name__ =='__main__':
    main()