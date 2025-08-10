
import torch

def seq_to_mtp(
    targets: torch.Tensor,
    n_future_tokens: int
) -> torch.Tensor:
    """
    Generates a tensor of future targets on the fly from a standard target tensor.

    Args:
        targets (torch.Tensor): The standard target tensor, shape (B, T).
        n_future_tokens (int): The number of future tokens to predict for each time step.

    Returns:
        torch.Tensor: The target tensor of shape (B, T, n_future_tokens).
                      y[b, t, k] corresponds to the (k+1)-th token after input_ids[b, t].
    """
    B, T = targets.shape
    all_targets = []
    for i in range(n_future_tokens):
        # Get the i-th future token for each position by rolling the tensor
        future_targets = torch.roll(targets, shifts=-i, dims=1)
        # The values that "wrap around" from the beginning of the sequence are invalid.
        # We replace them with the ignore_index (-1)
        if i > 0:
            future_targets[:, -i:] = -1
        all_targets.append(future_targets)
    # Stack along the last dimension to create the (B, T, n_future_tokens) tensor
    return torch.stack(all_targets, dim=2)
