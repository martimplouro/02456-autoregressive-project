import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(
        self,
        vocab_size=23,
        embedding_dim=32,
        sequence_length=129,
        num_families=3,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        input_dim = sequence_length * embedding_dim

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            num_families
        )

    def forward(self, x):

        # Convert token IDs into embeddings
        x = self.embedding(x)

        # Flatten sequence and embedding dimensions
        x = x.reshape(x.shape[0], -1)

        # Hidden layer
        x = F.relu(self.fc1(x))

        # Output: 3 family logits
        logits = self.fc2(x)

        return logits


# Test model
model = Classifier()

x = torch.randint(0, 23, (4, 129))

logits = model(x)

print(logits.shape)
