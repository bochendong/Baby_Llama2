import torch
import math
from torch import nn
from typing import Tuple

def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes frequency tensors for broadcasting with the input tensor x.

    Args:
        freqs (torch.Tensor): The frequency tensor, either cosine or sine.
        x (torch.Tensor): The tensor to which the frequencies will be applied.

    Returns:
        torch.Tensor: Reshaped frequency tensor for broadcasting.
    """
    # Ensure compatibility in dimensions
    assert 0 <= 1 < x.ndim, "x must have at least two dimensions."
    assert freqs.shape == (x.shape[1], x.shape[-1]), "Frequency shapes must match specific dimensions of x."
    
    # Prepare shape for broadcasting
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embedding to query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor.
        xk (torch.Tensor): Key tensor.
        freqs_cos (torch.Tensor): Cosine frequencies for rotation.
        freqs_sin (torch.Tensor): Sine frequencies for rotation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of transformed query and key tensors.
    """
    # Split last dimension into real and imaginary parts for complex number representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape frequency tensors for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply the rotary embedding
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten the last two dimensions back into one
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, n_heads, embed_dim, dropout):
        """
        Initializes the Attention module.

        Args:
            n_heads (int): Number of attention heads.
            embed_dim (int): Embedding dimension.
            dropout (float): Dropout rate.
        """
        super(Attention, self).__init__()
        self.num_heads = n_heads
        self.dropout = dropout
        self.head_dim = embed_dim // n_heads
        self.scaling_factor = math.sqrt(self.head_dim)
        
        # Linear transformations for queries, keys, and values
        self.wq = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin):
        """
        Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cos (torch.Tensor): Cosine frequencies for rotation.
            freqs_sin (torch.Tensor): Sine frequencies for rotation.

        Returns:
            torch.Tensor: Output tensor after attention.
        """
        bsz, seqlen, _ = x.shape

        # Project input x to queries (xq), keys (xk), and values (xv)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # Transpose for attention operation: from (batch, sequence, heads, features)
        # to (batch, heads, sequence, features) for easier manipulation
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        output, _ = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0)
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output