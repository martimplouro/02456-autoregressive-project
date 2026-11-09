import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(
        self,
        vocab_size=23, #numero de aminoacidos que podemos prever mais o start, end e o padding
        embedding_dim=32, #tem a ver com embedding
        family_embedding_dim=8,
        num_families=3,
        context_size=3, #numero de aminoacidos que vamos usar para prever o proximo
        hidden_dim=128 #numero de neuronios
    ):
        super().__init__()

        self.context_size = context_size

        self.embedding = nn.Embedding( #o embedding faz com que a nn nao use relacoes numericas entre os tokens dos aminoacidos
            vocab_size,                #tipo 3(A) < 4(C)
            embedding_dim,
            padding_idx=0
        )

        self.family_embedding = nn.Embedding(
            num_families,
            family_embedding_dim
        )

        input_dim = context_size * embedding_dim + family_embedding_dim

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )   

        self.fc2 = nn.Linear(
            hidden_dim, vocab_size
        )

    def forward(self, x, family):

        x = self.embedding(x)

        x = x.reshape(x.shape[0], -1)

        family = self.family_embedding(family)

        x = torch.cat((x, family), dim=1)

        x = F.relu(self.fc1(x))

        logits = self.fc2(x)

        return logits


model = MLP()
x = torch.randint(1, 23, (4, 3))
family = torch.randint(0, 3, (4,))
logits = model(x, family)
print(logits.shape)  # expect torch.Size([4, 23])
