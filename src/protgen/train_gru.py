
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models import GRULanguageModel


# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

VOCAB_SIZE = 23
NUM_FAMILIES = 3
IGNORE_INDEX = -1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# Load datasets
train_dataset = ProteinDataset("data/train.csv")
val_dataset = ProteinDataset("data/val.csv")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Initialise model
model = GRULanguageModel(
    vocab_size=VOCAB_SIZE,
    num_families=NUM_FAMILIES,
    embedding_dim=64,
    family_embedding_dim=8,
    hidden_dim=128,
    num_layers=2,
    dropout=0.15
).to(DEVICE)


# Loss and optimiser
criterion = nn.CrossEntropyLoss(
    ignore_index=IGNORE_INDEX
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Store losses
train_loss_save = []
val_loss_save = []


# Training
for epoch in range(NUM_EPOCHS):

    model.train()

    total_train_loss = 0
    total_train_tokens = 0

    for x, x_shifted, y, length in train_loader:

        x = x.to(DEVICE)
        x_shifted = x_shifted.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass, including family
        logits = model(x, y)

        # Flatten predictions and targets
        logits = logits.reshape(-1, VOCAB_SIZE)
        targets = x_shifted.reshape(-1)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        valid_tokens = (targets != IGNORE_INDEX).sum().item()

        total_train_loss += loss.item() * valid_tokens
        total_train_tokens += valid_tokens

    train_loss = total_train_loss / total_train_tokens
    train_loss_save.append(train_loss)


    # Validation
    model.eval()

    total_val_loss = 0
    total_val_tokens = 0

    with torch.no_grad():

        for x, x_shifted, y, length in val_loader:

            x = x.to(DEVICE)
            x_shifted = x_shifted.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x, y)

            logits = logits.reshape(-1, VOCAB_SIZE)
            targets = x_shifted.reshape(-1)

            loss = criterion(logits, targets)

            valid_tokens = (targets != IGNORE_INDEX).sum().item()

            total_val_loss += loss.item() * valid_tokens
            total_val_tokens += valid_tokens

    val_loss = total_val_loss / total_val_tokens
    val_loss_save.append(val_loss)


    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Validation Loss: {val_loss:.4f}"
    )


# Save model
torch.save(
    model.state_dict(),
    "gru_model.pth"
)


# Save loss history
losses_df = pd.DataFrame({
    "epoch": range(1, NUM_EPOCHS + 1),
    "train_loss": train_loss_save,
    "val_loss": val_loss_save
})

losses_df.to_csv("gru_training_losses.csv", index=False)

print("Training finished. GRU model saved!")
