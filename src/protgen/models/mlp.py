import torch
import torch.nn.functional as F

def generate(model, max_length=127, temperature=1.0):

    model.eval()

    device = next(model.parameters()).device

    sequence = torch.tensor(
        [[1]],  # BOS
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_length):

            # Use only the previous context_size tokens
            context = sequence[:, -model.context_size:]

            # Pad context on the left if necessary
            if context.shape[1] < model.context_size:
                context = F.pad(
                    context,
                    (model.context_size - context.shape[1], 0),
                    value=0
                )

            logits_gen, _ = model(context)

            # Predictions at the last position
            logits = logits_gen[:, -1, :]

            # Temperature controls sampling randomness
            probabilities = F.softmax(
                logits / temperature,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            if next_token.item() == 2:  # EOS
                break

            sequence = torch.cat(
                [sequence, next_token],
                dim=1
            )

    # Convert token IDs back into amino acids
    return sequence