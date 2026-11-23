
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models import Classifier

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

VOCAB_SIZE = 23
NUM_FAMILIES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# Load datasets
train_dataset = ProteinDataset("data/train.csv")
val_dataset = ProteinDataset("data/val.csv")
test_dataset = ProteinDataset("data/test.csv")

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Initialise classifier
model = Classifier(
    vocab_size=VOCAB_SIZE,
    num_families=NUM_FAMILIES,
    sequence_length=129
).to(DEVICE)

# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

train_loss_save = []
val_loss_save = []

best_val_loss = float("inf")

# Training and validation
for epoch in range(NUM_EPOCHS):

    model.train()

    total_train_loss = 0
    total_train_samples = 0
    correct_train = 0

    for x, x_shifted, family, length in train_loader:

        x = x.to(DEVICE)
        family = family.to(DEVICE).long()

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, family)

        loss.backward()
        optimizer.step()

        batch_size = family.size(0)

        total_train_loss += loss.item() * batch_size
        total_train_samples += batch_size

        predictions = logits.argmax(dim=1)
        correct_train += (predictions == family).sum().item()

    train_loss = total_train_loss / total_train_samples
    train_accuracy = correct_train / total_train_samples

    train_loss_save.append(train_loss)

    model.eval()

    total_val_loss = 0
    total_val_samples = 0
    correct_val = 0

    with torch.no_grad():

        for x, x_shifted, family, length in val_loader:

            x = x.to(DEVICE)
            family = family.to(DEVICE).long()

            logits = model(x)

            loss = criterion(logits, family)

            batch_size = family.size(0)

            total_val_loss += loss.item() * batch_size
            total_val_samples += batch_size

            predictions = logits.argmax(dim=1)
            correct_val += (predictions == family).sum().item()

    val_loss = total_val_loss / total_val_samples
    val_accuracy = correct_val / total_val_samples

    val_loss_save.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_classifier.pth"
        )

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )

model.load_state_dict(
    torch.load("best_classifier.pth", map_location=DEVICE)
)

model.eval()

total_test_loss = 0
total_test_samples = 0
correct_test = 0

with torch.no_grad():

    for x, x_shifted, family, length in test_loader:

        x = x.to(DEVICE)
        family = family.to(DEVICE).long()

        logits = model(x)

        loss = criterion(logits, family)

        batch_size = family.size(0)

        total_test_loss += loss.item() * batch_size
        total_test_samples += batch_size

        predictions = logits.argmax(dim=1)
        correct_test += (predictions == family).sum().item()

test_loss = total_test_loss / total_test_samples
test_accuracy = correct_test / total_test_samples

print("\nTEST RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

losses_df = pd.DataFrame({
    "epoch": range(1, NUM_EPOCHS + 1),
    "train_loss": train_loss_save,
    "val_loss": val_loss_save
})

losses_df.to_csv(
    "classifier_training_losses.csv",
    index=False
)