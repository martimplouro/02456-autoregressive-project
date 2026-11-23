
import torch
import torch.nn as nn


class GRULanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=23,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.15
    ):
        super().__init__()

        # Amino acid embedding
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # GRU receives only amino acid embeddings
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Predict next amino acid
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        output, hidden = self.gru(x)

        logits = self.fc(output)

        return logits