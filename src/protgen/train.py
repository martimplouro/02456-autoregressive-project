
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models import MLP

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
CONTEXT_SIZE = 3

VOCAB_SIZE = 23
NUM_FAMILIES = 3

PAD_TOKEN = 0
IGNORE_INDEX = -1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


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



model = MLP(
    vocab_size=VOCAB_SIZE,
    num_families=NUM_FAMILIES,
    embedding_dim=32,
    family_embedding_dim=8,
    context_size=CONTEXT_SIZE,
    hidden_dim=128
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def prepare_batch(x, x_shifted, family):

    # Add padding to the left of each sequence
    x_padded = torch.nn.functional.pad(
        x,
        (CONTEXT_SIZE - 1, 0),
        value=PAD_TOKEN
    )

    # Extract sliding windows of tokens
    contexts = x_padded.unfold(
        dimension=1,
        size=CONTEXT_SIZE,
        step=1
    )

    # Flatten batch and sequence dimensions
    contexts = contexts.reshape(-1, CONTEXT_SIZE)
    targets = x_shifted.reshape(-1)

    # Repeat family labels for every token
    families = family.unsqueeze(1).expand(
        -1, x.shape[1]
    ).reshape(-1)

    # Ignore positions without a valid target
    mask = targets != IGNORE_INDEX

    return contexts[mask], targets[mask], families[mask]



for epoch in range(NUM_EPOCHS):

    model.train()

    total_train_loss = 0
    total_train_tokens = 0

    for x, x_shifted, y, length in train_loader:

        x = x.to(DEVICE)
        x_shifted = x_shifted.to(DEVICE)
        y = y.to(DEVICE)

        # Prepare model inputs and targets
        contexts, targets, families = prepare_batch(
            x, x_shifted, y
        )

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(contexts, families)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item() * targets.numel()
        total_train_tokens += targets.numel()

    train_loss = total_train_loss / total_train_tokens


    model.eval()

    total_val_loss = 0
    total_val_tokens = 0

    with torch.no_grad():

        for x, x_shifted, y, length in val_loader:

            x = x.to(DEVICE)
            x_shifted = x_shifted.to(DEVICE)
            y = y.to(DEVICE)

            contexts, targets, families = prepare_batch(
                x, x_shifted, y
            )

            logits = model(contexts, families)

            loss = criterion(logits, targets)

            total_val_loss += loss.item() * targets.numel()
            total_val_tokens += targets.numel()

    val_loss = total_val_loss / total_val_tokens


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Validation Loss: {val_loss:.4f}"
    )



torch.save(
    model.state_dict(),
    "mlp_model.pth"
)

print("Training finished. Model saved!")
