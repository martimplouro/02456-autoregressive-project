
import torch
import torch.nn as nn


class GRULanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=20, #number of aminoacids taken
        embedding_dim=64, #dimension of each aminoacid embedding, the size that each vector will have
        hidden_dim=128, #number of GRU hidden layers 
        num_layers=2, #number of layers in the model
        dropout=0.15 #reducing overfitting that might happen
    ):
        super().__init__()

        # Convert amino acid tokens into vectors
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        # Recurrent neural network
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Predict the next amino acid
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        # x: [batch_size, sequence_length]

        embedded = self.embedding(x)

        # GRU processes the sequence
        output, hidden = self.gru(embedded)

        # Predict amino acid probabilities (as logits)
        logits = self.fc(output)

        return logits
