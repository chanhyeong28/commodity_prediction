import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from typing import Optional, Tuple, Union
from einops import rearrange


class TriangularCausalMask:
    """Triangular causal mask for autoregressive attention."""
    def __init__(self, B: int, L: int, device: str = "cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Probabilistic attention mask for sparse attention."""
    def __init__(self, B: int, H: int, L: int, index: torch.Tensor, scores: torch.Tensor, device: str = "cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    Standard full self-attention mechanism.
    Compatible with both Time-LLM and iTransformer implementations.
    """
    def __init__(
        self, 
        mask_flag: bool = True, 
        factor: int = 5, 
        scale: Optional[float] = None, 
        attention_dropout: float = 0.1, 
        output_attention: bool = False
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for full attention.
        
        Args:
            queries: [B, L, H, E] - Query tensor
            keys: [B, S, H, E] - Key tensor  
            values: [B, S, H, D] - Value tensor
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: [B, L, H, D] - Attention output
            attention_weights: Optional attention weights
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Compute attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply de-stationary factors if provided
        if tau is not None:
            tau = tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
            scores = scores * tau
        if delta is not None:
            delta = delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S
            scores = scores + delta

        # Apply causal mask if enabled
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Apply softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    Probabilistic sparse attention mechanism from Informer.
    Reduces computational complexity by selecting top-k queries.
    """
    def __init__(
        self, 
        mask_flag: bool = True, 
        factor: int = 5, 
        scale: Optional[float] = None, 
        attention_dropout: float = 0.1, 
        output_attention: bool = False
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample top-k queries based on sparsity measurement."""
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Find top-k queries with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """Initialize context for attention computation."""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert (L_Q == L_V), "For masked attention, L_Q must equal L_V"
            context = V.cumsum(dim=-2)
        return context

    def _update_context(
        self, 
        context_in: torch.Tensor, 
        V: torch.Tensor, 
        scores: torch.Tensor, 
        index: torch.Tensor, 
        L_Q: int, 
        attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Update context with selected top-k queries."""
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for probabilistic attention."""
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class FlashAttention(nn.Module):
    """
    Memory-efficient Flash Attention implementation.
    Reduces memory usage through block-wise computation.
    """
    def __init__(
        self, 
        mask_flag: bool = True, 
        factor: int = 5, 
        scale: Optional[float] = None, 
        attention_dropout: float = 0.1, 
        output_attention: bool = False,
        block_size: int = 32
    ):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.block_size = block_size

    def flash_attention_forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Block-wise flash attention forward pass."""
        NEG_INF = -1e10
        EPSILON = 1e-10
        
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device=Q.device)
        l = l.to(device=Q.device)
        m = m.to(device=Q.device)

        Q_BLOCK_SIZE = min(self.block_size, Q.shape[-1])
        KV_BLOCK_SIZE = self.block_size

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for flash attention."""
        res = self.flash_attention_forward(
            queries.permute(0, 2, 1, 3), 
            keys.permute(0, 2, 1, 3), 
            values.permute(0, 2, 1, 3),
            attn_mask
        )[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FlowAttention(nn.Module):
    """
    Flow Attention mechanism for efficient attention computation.
    Based on the Flowformer implementation.
    """
    def __init__(self, attention_dropout: float = 0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x: torch.Tensor) -> torch.Tensor:
        """Apply kernel function to input."""
        return torch.sigmoid(x)

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for flow attention."""
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Apply kernel function
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
        # Compute normalizers
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        
        # Refine normalizers
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        
        # Competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]
        
        # Compute output
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1, 2).contiguous()
        
        return x, None


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention layer.
    Compatible with Time-LlaMA's channel-as-token approach.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
        attention_type: str = 'full',
        **attention_kwargs
    ):
        super(MultiHeadAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        
        # Initialize attention mechanism
        if attention_type == 'full':
            self.inner_attention = FullAttention(**attention_kwargs)
        elif attention_type == 'prob':
            self.inner_attention = ProbAttention(**attention_kwargs)
        elif attention_type == 'flash':
            self.inner_attention = FlashAttention(**attention_kwargs)
        elif attention_type == 'flow':
            self.inner_attention = FlowAttention(**attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [self.query_projection, self.key_projection, self.value_projection, self.out_projection]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None, 
        tau: Optional[torch.Tensor] = None, 
        delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            queries: [B, L, d_model] - Query tensor
            keys: [B, S, d_model] - Key tensor
            values: [B, S, d_model] - Value tensor
            attn_mask: Optional attention mask
            tau: Optional de-stationary factor
            delta: Optional de-stationary factor
            
        Returns:
            output: [B, L, d_model] - Attention output
            attention_weights: Optional attention weights
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project to multi-head format
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply attention
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        
        # Reshape and project output
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for Time-LlaMA's prompt alignment.
    Enables interaction between time series tokens and text prompt embeddings.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_keys: Optional[int] = None,
        d_values: Optional[int] = None,
        attention_dropout: float = 0.1
    ):
        super(CrossAttention, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_values = d_values
        
        # Projection layers
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / sqrt(d_keys)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [self.query_projection, self.key_projection, self.value_projection, self.out_projection]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        
        Args:
            queries: [B, N, d_model] - Time series tokens (queries)
            keys: [B, P, d_model] - Prompt embeddings (keys)
            values: [B, P, d_model] - Prompt embeddings (values)
            
        Returns:
            output: [B, N, d_model] - Aligned time series tokens
        """
        B, N, _ = queries.shape
        _, P, _ = keys.shape
        H = self.n_heads

        # Project to multi-head format
        Q = self.query_projection(queries).view(B, N, H, -1)
        K = self.key_projection(keys).view(B, P, H, -1)
        V = self.value_projection(values).view(B, P, H, -1)

        # Compute attention scores
        scores = torch.einsum("bnhd,bphd->bhnp", Q, K) * self.scale
        
        # Apply softmax and dropout
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        
        # Apply attention to values
        out = torch.einsum("bhnp,bphd->bnhd", attn_weights, V)
        
        # Reshape and project output
        out = out.view(B, N, -1)
        return self.out_projection(out)
