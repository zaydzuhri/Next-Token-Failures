import torch

def seq_to_dsmtp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model_seq_len: int,
    n_future_tokens: int
) -> torch.Tensor:
    # pad labels to the right with -1s to handle the last few tokens
    input_ids = torch.nn.functional.pad(input_ids, (0, n_future_tokens), value=1)
    labels = torch.nn.functional.pad(labels, (1, n_future_tokens-1), value=-1)
    B, total_len = labels.shape # (B, T)

    windows = labels.unfold(dimension=1, size=n_future_tokens + 1, step=1) # (B, T - n_future_tokens, n_future_tokens + 1)
    input_windows = input_ids.unfold(dimension=1, size=n_future_tokens + 1, step=1) # (B, T - n_future_tokens, n_future_tokens + 1)

    all_targets = windows[:, :, 1:] # (B, T - n_future_tokens, n_future_tokens)
    output_targets = all_targets

    all_inputs = input_windows[:, :, :-1] # (B, T - n_future_tokens, n_future_tokens)
    all_inputs = all_inputs[:, :model_seq_len, :] # (B, model_seq_len, n_future_tokens)
    return all_inputs.transpose(1, 2), output_targets.transpose(1, 2)