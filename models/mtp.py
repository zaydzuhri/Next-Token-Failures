import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch_compile_options = {
    'epilogue_fusion': True,
    'max_autotune': False,
    'shape_padding': True,
    'trace.enabled': False,
    'triton.cudagraphs': False,
    'debug': False,
    'dce': True,
    'memory_planning': True,
    'coordinate_descent_tuning': False,
    'trace.graph_diagram': False,
    'compile_threads': 32,
    'group_fusion': True,
    'disable_progress': True,
    'verbose_progress': False,
    'cuda.compile_opt_level': '-O2',
    'cuda.enable_cuda_lto': True
}

@torch.compile(dynamic=True, fullgraph=True, options=torch_compile_options)
def seq_to_mtp(
    long_input_ids: torch.Tensor,
    model_seq_len: int,
    n_future_tokens: int
) -> torch.Tensor:
    """
    Generates a tensor of future targets on the fly from a long input sequence.

    This version assumes `long_input_ids` contains both the tokens for the model's
    input AND the future tokens needed for the labels.
    It extracts the correct targets without adding artificial padding.

    Args:
        long_input_ids (torch.Tensor): The input sequences from the dataloader,
                                       shape (B, T + n_future_tokens).
        model_seq_len (int): The sequence length `T` that the model processes.
        n_future_tokens (int): The number of future tokens to predict for each time step.

    Returns:
        torch.Tensor: The target tensor of shape (B, T, n_future_tokens).
                      y[b, t, k] corresponds to the (k+1)-th token after input_ids[b, t].
    """
    B, total_len = long_input_ids.shape
    assert total_len >= model_seq_len + n_future_tokens, \
        "long_input_ids must be at least model_seq_len + n_future_tokens long."

    # 1. Create sliding windows (views) over the long tensor.
    # .unfold() is a highly efficient way to create sliding windows.
    # We create windows of size `n_future_tokens + 1`. For each time step `t`,
    # the window will contain the input token and its `n_future_tokens` targets.
    # Example (n=3, window_size=4):
    # For t=0, window is [t0, t1, t2, t3]
    # For t=1, window is [t1, t2, t3, t4]
    # Shape of windows: (B, total_len - n_future_tokens, n_future_tokens + 1)
    windows = long_input_ids.unfold(dimension=1, size=n_future_tokens + 1, step=1)

    # 2. Slice the windows to get only the targets.
    # We slice off the first element of each window (the input token itself)
    # to keep only the future tokens.
    # Example window [t0, t1, t2, t3] -> becomes targets [t1, t2, t3]
    all_targets = windows[:, :, 1:]

    # 3. Trim the result to match the model's output sequence length.
    # We only need the targets for the first `model_seq_len` positions.
    output_targets = all_targets[:, :model_seq_len, :]

    return output_targets.transpose(1, 2)

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['D']
)
@triton.jit
def logsumexp_fwd_kernel(
    x,
    z,
    scale,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


def logsumexp_fwd(
    x,
    scale: Optional[float] = None,
    dtype: Optional[torch.dtype] = None
):
    r"""
    Compute the logsumexp of the input tensor over the last dimension.

    Args:
        x (Tensor):
            The input tensor of any shape.
        scale (Optional[float]):
            The scale applied to the input tensor. Default: `None`.
        dtype (Optional[torch.dtype]):
            The data type of the output tensor. Default: `None`.
    Returns:
        Tensor: The logsumexp of the input tensor.
    """

    shape = x.shape
    x = x.view(-1, shape[-1])
    N, D = x.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    z = x.new_empty(N, ND, dtype=torch.float)
    logsumexp_fwd_kernel[(N, ND)](
        x=x,
        z=z,
        scale=scale,
        D=D,
        B=B
    )
    z = z.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        z = z.to(dtype)
    return z


# -*- coding: utf-8 -*-

# Code adapted from
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2

@triton.jit
def listnet_kernel(
    logits,
    targets,  # Now full target distributions
    lse_logits,
    lse_targets,
    loss,
    total,
    ignore_index,
    logit_scale: tl.constexpr,
    reduction: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr
):
    i_n = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    
    # Pointers to current token's data
    logits_ptr = logits + i_n * V
    targets_ptr = targets + i_n * V
    loss_ptr = loss + i_n
    
    # Compute prediction softmax
    b_lse_logits = tl.load(lse_logits + i_n)
    b_lse_targets = tl.load(lse_targets + i_n)
    b_loss = 0.0
    
    # Compute gradient: softmax(pred) - softmax(target)
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        mask = o_v < V
        
        # Load target and compute softmax
        t_val = tl.load(targets_ptr + o_v, mask=mask, other=0.0)
        p_target = tl.exp(t_val - b_lse_targets)
        
        # Load logits and compute softmax
        l_val = tl.load(logits_ptr + o_v, mask=mask, other=0.0) * logit_scale
        l_val_minus_lse = l_val - b_lse_logits
        p_pred = tl.exp(l_val_minus_lse)
        
        # Gradient calculation
        grad_val = p_pred - p_target
        if reduction == "mean":
            grad_val = grad_val / total
        grad_val = tl.where(b_lse_targets == float('inf'), 0.0, grad_val)
        tl.store(logits_ptr + o_v, grad_val, mask=mask)
        
        # Cross-entropy loss
        # instead of: b_loss -= tl.sum(p_target * tl.log(p_pred), axis=0)
        b_loss -= tl.sum(p_target * l_val_minus_lse, axis=0)
    
    tl.store(loss_ptr, b_loss)

@triton.jit
def elementwise_mul_kernel(
    x,
    g,
    N: tl.constexpr,
    B: tl.constexpr
):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    i_x = tl.program_id(0).to(tl.int64)
    o_x = i_x * B + tl.arange(0, B)

    # Load the gradient output value
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


def fused_linear_listnet_forward(
    x: torch.Tensor,
    targets: torch.Tensor,  # Float tensor [N, V]
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean"
):
    N, H, V = *x.shape, weight.shape[0]
    BV = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    NC = min(num_chunks, triton.cdiv(V, H))
    C = triton.next_power_of_2(triton.cdiv(N, NC))
    NC = triton.cdiv(N, C)

    # Initialize outputs
    dx = torch.zeros_like(x)
    dw = torch.zeros_like(weight, dtype=torch.float) if weight is not None else None
    db = torch.zeros_like(bias, dtype=torch.float) if bias is not None else None
    loss = torch.zeros(N, device=x.device, dtype=torch.float)
    total = N  # All tokens considered

    for ic in range(NC):
        start, end = ic * C, min((ic + 1) * C, N)
        c_x = x[start:end]
        c_logits = F.linear(c_x, weight, bias)
        c_targets = targets[start:end]
        c_lse_logits = logsumexp_fwd(c_logits, scale=logit_scale, dtype=torch.float)
        c_lse_targets = logsumexp_fwd(c_targets, dtype=torch.float).nan_to_num(nan=float("inf"))
        c_loss = loss[start:end]

        # Call ListNet kernel
        listnet_kernel[(c_logits.shape[0],)](
            logits=c_logits,
            targets=c_targets,  # Full target distributions
            lse_logits=c_lse_logits,
            lse_targets=c_lse_targets,
            loss=c_loss,
            total=total,
            ignore_index=ignore_index,
            logit_scale=logit_scale,
            reduction=reduction,
            V=V,
            BV=BV,
            num_warps=32
        )

        # Backward through linear layer
        dx[start:end] = torch.mm(c_logits, weight)
        if weight is not None:
            dw += c_logits.t() @ c_x
        if bias is not None:
            db += c_logits.sum(0)

    loss = loss.sum()
    if reduction == "mean":
        loss = loss / total
        
    return loss, dx, dw, db


def fused_linear_listnet_backward(
    do: torch.Tensor,
    dx: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor
):
    # If cross entropy is the last layer, do is 1.0. Skip the mul to save time
    if torch.ne(do, torch.tensor(1.0, device=do.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        N, H = dx.shape
        B = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        elementwise_mul_kernel[(triton.cdiv(N * H, B),)](
            x=dx,
            g=do,
            N=N*H,
            B=B,
            num_warps=32,
        )

        # handle dw
        if dw is not None:
            V, H = dw.shape
            elementwise_mul_kernel[(triton.cdiv(V * H, B),)](
                x=dw,
                g=do,
                N=V*H,
                B=B,
                num_warps=32,
            )

        if db is not None:
            V = db.shape[0]
            elementwise_mul_kernel[(triton.cdiv(V, B),)](
                x=db,
                g=do,
                N=V,
                B=B,
                num_warps=32,
            )
    return dx, dw, db


class FusedLinearListNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        targets: torch.Tensor,  # Float targets
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        ignore_index: int = -100,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean"
    ):
        loss, dx, dw, db = fused_linear_listnet_forward(
            x, targets, weight, bias, ignore_index, 
            logit_scale, num_chunks, reduction
        )
        ctx.save_for_backward(dx, dw, db)
        return loss

    @staticmethod
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        dx, dw, db = fused_linear_listnet_backward(do, dx, dw, db)
        return dx, None, dw, db, None, None, None, None


def fused_linear_listnet_loss(
    x: torch.Tensor,
    target: torch.LongTensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    num_chunks: int = 8,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): [batch_size * seq_len, hidden_size]
        target (torch.LongTensor): [batch_size * seq_len]
            where each value is in [0, vocab_size).
        weight (torch.Tensor): [vocab_size, hidden_size]
            where `vocab_size` is the number of classes.
        bias (Optional[torch.Tensor]): [vocab_size]
            where `vocab_size` is the number of classes.
        ignore_index: int.
            If target == ignore_index, the loss is set to 0.0.
        label_smoothing: float
        logit_scale: float
            A scaling factor applied to the logits. Default: 1.0
        num_chunks: int
            The number of chunks to split the input tensor into for processing.
            This can help optimize memory usage and computation speed.
            Default: 8
        reduction:
            Specifies the reduction to apply to the output: 'mean' | 'sum'.
            'mean': the weighted mean of the output is taken,
            'sum': the output will be summed.
            Default: 'mean'.
    Returns:
        losses: [batch,], float
    """
    return FusedLinearListNetFunction.apply(
        x,
        target,
        weight,
        bias,
        ignore_index,
        logit_scale,
        num_chunks,
        reduction
    )


class FusedLinearListNetLoss(nn.Module):

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        logit_scale: float = 1.0,
        num_chunks: int = 8,
        reduction: str = "mean"
    ):
        """
        Args:
            ignore_index: int.
                If target == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            logit_scale: float
                A scaling factor applied to the logits. Default: 1.0
            num_chunks: int
                The number of chunks to split the input tensor into for processing.
                This can help optimize memory usage and computation speed.
                Default: 8
            reduction:
                Specifies the reduction to apply to the output: 'mean' | 'sum'.
                'mean': the weighted mean of the output is taken,
                'sum': the output will be summed.
                Default: 'mean'.
        """
        super().__init__()

        assert reduction in ["mean", "sum"], f"reduction: {reduction} is not supported"

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.num_chunks = num_chunks
        self.reduction = reduction

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        target: torch.LongTensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
            target (torch.LongTensor): [batch_size, seq_len]
                where each value is in [0, V).
            weight (torch.Tensor): [vocab_size, hidden_size]
                where `vocab_size` is the number of classes.
            bias (Optional[torch.Tensor]): [vocab_size]
                where `vocab_size` is the number of classes.
        Returns:
            loss
        """
        loss = fused_linear_listnet_loss(
            x.view(-1, x.shape[-1]),
            target.view(-1, target.shape[-1]),
            weight=weight,
            bias=bias,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            num_chunks=self.num_chunks,
            reduction=self.reduction
        )
        return loss

# Naive ListNet loss function implementation
def list_net_loss(y_pred, y_true):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [*, slate_length]
    :param y_true: ground truth labels, shape [*, slate_length]
    :return: loss value, a torch.Tensor
    """
    return torch.mean(-torch.sum(F.softmax(y_true, dim=-1).nan_to_num(nan=0) * F.log_softmax(y_pred, dim=-1), dim=-1))