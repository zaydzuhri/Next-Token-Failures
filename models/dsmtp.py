import torch

def seq_to_dsmtp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model_seq_len: int,
    n_future_tokens: int
) -> torch.Tensor:
    B, total_len = labels.shape
    assert total_len >= model_seq_len + n_future_tokens, \
        "long_input_ids must be at least model_seq_len + n_future_tokens long."

    windows = labels.unfold(dimension=1, size=n_future_tokens + 1, step=1)

    all_targets = windows[:, :, 1:]

    output_targets = all_targets[:, :model_seq_len, :]

    all_inputs = all_targets[:, :model_seq_len, :-1]
    all_inputs = torch.concat([input_ids.unsqueeze(-1), all_inputs], dim=-1)

    return all_inputs.transpose(1, 2), output_targets.transpose(1, 2)