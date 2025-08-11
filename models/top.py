import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_V': block_size_v}, num_warps=num_warp)
        for block_size_v in [256, 512, 1024, 2048]
        for num_warp in [1, 2, 4, 8]
    ],
    key=['V'],
)
@triton.jit
def _seq_to_top_kernel(
    seq_ptr,
    output_ptr,
    B,
    T_total,
    T,
    V,
    pad_token_id,
    window_size,
    T_val,
    stride_seq_b,
    stride_seq_t,
    stride_out_b,
    stride_out_t,
    stride_out_v,
    BLOCK_SIZE_V: tl.constexpr,
):
    b = tl.program_id(0)
    v_block = tl.program_id(1)
    
    v_start = v_block * BLOCK_SIZE_V
    v_end = tl.minimum(v_start + BLOCK_SIZE_V, V)
    v_idx = tl.arange(0, BLOCK_SIZE_V)
    v = v_start + v_idx
    mask = v < V
    
    next_occurrence = tl.full((BLOCK_SIZE_V,), T_val, dtype=tl.int64)
    
    for t in range(T_total - 1, -1, -1):
        token = tl.load(seq_ptr + b * stride_seq_b + t * stride_seq_t)
        
        token_valid = (token != pad_token_id) & (token >= 0) & (token < V)
        in_block = (token >= v_start) & (token < v_end)
        
        if token_valid:
            if in_block:
                local_v = token - v_start
                next_occurrence = tl.where(v_idx == local_v, t, next_occurrence)
        
        if t < T:
            distance = next_occurrence - t
            valid = (distance < window_size)
            value = tl.where(valid, window_size - distance, float('-inf'))
            
            output_offset = (
                b * stride_out_b +
                t * stride_out_t +
                v * stride_out_v
            )
            tl.store(output_ptr + output_offset, value, mask=mask)

def seq_to_top(
    seq: torch.Tensor, 
    vocab_size: int, 
    window_size: int,
    pad_token_id: int = -100
) -> torch.Tensor:
    """
    Triton-optimized top sequence processing with autotuned block size.
    
    :param seq: Input sequence of shape (B, T + window_size)
    :param vocab_size: Size of the vocabulary
    :param window_size: How far ahead to look for next occurrences
    :param pad_token_id: Padding token ID
    :return: Tensor of shape (B, T, V) with window_size - distance for tokens in window, else -inf
    """
    B, T_total = seq.shape
    T = T_total - window_size
    
    assert T > 0, "T_total must be greater than window_size to produce valid output."
    
    output = torch.empty((B, T, vocab_size), device=seq.device, dtype=torch.float16)
    if not output.is_contiguous():
        output = output.contiguous()
    
    # Let autotune select the best BLOCK_SIZE_V based on vocab_size
    grid = (B, triton.cdiv(vocab_size, 128))  # Start with minimum block size
    
    _seq_to_top_kernel[grid](
        seq,
        output,
        B,
        T_total,
        T,
        vocab_size,
        pad_token_id,
        window_size,
        T_total,
        seq.stride(0),
        seq.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
    )
    
    return output

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