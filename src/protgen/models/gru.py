
import torch
import torch.nn as nn


class GRULanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=23,
        num_families=3,
        embedding_dim=64,
        family_embedding_dim=8,
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

        # Protein family embedding
        self.family_embedding = nn.Embedding(
            num_families,
            family_embedding_dim
        )

        # GRU receives amino acid + family embeddings
        self.gru = nn.GRU(
            input_size=embedding_dim + family_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Predict next amino acid
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, family):

        # x: (B, sequence_length)
        # family: (B,)

        # Amino acid embeddings
        x = self.embedding(x)
        # (B, sequence_length, embedding_dim)

        # Family embedding
        family = self.family_embedding(family)
        # (B, family_embedding_dim)

        # Repeat family embedding at every sequence position
        family = family.unsqueeze(1).expand(
            -1, x.shape[1], -1
        )
        # (B, sequence_length, family_embedding_dim)

        # Concatenate amino acid and family embeddings
        x = torch.cat((x, family), dim=2)
        # (B, sequence_length, embedding_dim + family_embedding_dim)

        # GRU
        output, hidden = self.gru(x)

        # Predict next token
        logits = self.fc(output)

        return logits
