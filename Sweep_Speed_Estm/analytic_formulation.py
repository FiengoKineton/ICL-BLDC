from dataclasses import dataclass
from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict

import math, torch
import torch.nn as nn, torch.nn.functional as F

######################################################################

@dataclass
class GPTConfig:
    block_size: int = 10     # T_max
    n_layer: int = 8         # L
    n_head: int = 8          # h
    n_embd: int = 32         # d
    n_u: int = 5             # input dim
    n_y: int = 1             # output dim
    dropout: float = 0.0     # p
    bias: bool = True        # biases in linear/LN
    ln_eps: float = 1e-5

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

@torch.no_grad()
def _assert_shapes(batch_u: torch.Tensor, batch_y: Optional[torch.Tensor] = None) -> None:
    assert batch_u.ndim == 3, "batch_u must be (B,T,n_u)"
    if batch_y is not None:
        assert batch_y.ndim == 3, "batch_y must be (B,T,n_y)"
        assert batch_u.shape[0] == batch_y.shape[0]
        assert batch_u.shape[1] == batch_y.shape[1]

######################################################################

class DecoderOnlyCausalTransformer(nn.Module):
    """
    Implements the exact math you wrote:
      U -> token embed -> + positional embed -> L x (Pre-LN Attn + Pre-LN MLP) -> final LN -> head
    with each operator implemented as a separate method inside the class.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "h must divide d"
        self.cfg = cfg
        self.d_h = cfg.n_embd // cfg.n_head

        # -------------------------
        # (1) Input embedding params: W_e, b_e, W_p
        # -------------------------
        # W_e in R^{d x n_u}, b_e in R^{d}
        self.W_e = nn.Parameter(torch.empty(cfg.n_embd, cfg.n_u))
        self.b_e = nn.Parameter(torch.empty(cfg.n_embd)) if cfg.bias else None

        # W_p in R^{T_max x d}  (we'll index first T rows at runtime)
        self.W_p = nn.Parameter(torch.empty(cfg.block_size, cfg.n_embd))

        # -------------------------
        # (2) Per-block parameters (all explicit, matching your symbols)
        # -------------------------
        self.blocks = nn.ModuleList([self._make_block_params() for _ in range(cfg.n_layer)])

        # -------------------------
        # (3) Final LN params: gamma_f, beta_f
        # -------------------------
        self.gamma_f = nn.Parameter(torch.ones(cfg.n_embd))
        self.beta_f  = nn.Parameter(torch.zeros(cfg.n_embd)) if cfg.bias else None

        # -------------------------
        # (4) Output head params: W_y, b_y
        # -------------------------
        self.W_y = nn.Parameter(torch.empty(cfg.n_y, cfg.n_embd))
        self.b_y = nn.Parameter(torch.empty(cfg.n_y)) if cfg.bias else None

        self.reset_parameters()

    def _make_block_params(self) -> nn.Module:
        """Creates one block's parameters as a small nn.Module container."""
        cfg = self.cfg
        d = cfg.n_embd

        block = nn.Module()

        # LN1 (gamma_1, beta_1), LN2 (gamma_2, beta_2)
        block.gamma_1 = nn.Parameter(torch.ones(d))
        block.beta_1  = nn.Parameter(torch.zeros(d)) if cfg.bias else None
        block.gamma_2 = nn.Parameter(torch.ones(d))
        block.beta_2  = nn.Parameter(torch.zeros(d)) if cfg.bias else None

        # Attn projections: W_qkv in R^{3d x d}, b_qkv in R^{3d}
        block.W_qkv = nn.Parameter(torch.empty(3 * d, d))
        block.b_qkv = nn.Parameter(torch.empty(3 * d)) if cfg.bias else None

        # Attn output: W_o in R^{d x d}, b_o in R^{d}
        block.W_o = nn.Parameter(torch.empty(d, d))
        block.b_o = nn.Parameter(torch.empty(d)) if cfg.bias else None

        # MLP: W_fc in R^{4d x d}, b_fc in R^{4d}
        block.W_fc = nn.Parameter(torch.empty(4 * d, d))
        block.b_fc = nn.Parameter(torch.empty(4 * d)) if cfg.bias else None

        # MLP proj: W_proj in R^{d x 4d}, b_proj in R^{d}
        block.W_proj = nn.Parameter(torch.empty(d, 4 * d))
        block.b_proj = nn.Parameter(torch.empty(d)) if cfg.bias else None

        return block

    def reset_parameters(self):
        """Simple init; you can swap to your preferred scheme."""
        def init_param(p):
            if p is None:
                return
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.zeros_(p)

        for p in [self.W_e, self.b_e, self.W_p, self.W_y, self.b_y, self.gamma_f, self.beta_f]:
            init_param(p)

        for blk in self.blocks:
            for name, p in blk.named_parameters():
                init_param(p)

        # It’s common to init gammas to 1
        with torch.no_grad():
            for blk in self.blocks:
                blk.gamma_1.fill_(1.0)
                blk.gamma_2.fill_(1.0)
            self.gamma_f.fill_(1.0)

    # =========================================================
    # Operator methods (each one separate, like you requested)
    # =========================================================

    def linear(self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        # x: (..., in), W: (out, in) => (..., out)
        y = x @ W.t()
        return y if b is None else (y + b)

    def ln(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor | None) -> torch.Tensor:
        # LN over last dimension d, applied independently over (B,T)
        eps = self.cfg.ln_eps
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        xhat = (x - mu) / torch.sqrt(var + eps)
        y = gamma * xhat
        return y if beta is None else (y + beta)

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        # p=0 => identity, but keep it explicit for completeness.
        p = self.cfg.dropout
        if (not self.training) or p == 0.0:
            return x
        return F.dropout(x, p=p, training=True)

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        # Matches definition; PyTorch gelu uses the same (exact or approximate depending).
        return F.gelu(x)

    def causal_mask(self, S: torch.Tensor, T: int) -> torch.Tensor:
        """
        S: (B, h, T, T)
        set S[..., t, j] = -inf if j > t
        """
        # Upper triangular (excluding diagonal) are future positions
        mask = torch.triu(torch.ones(T, T, device=S.device, dtype=torch.bool), diagonal=1)
        return S.masked_fill(mask, float("-inf"))

    def softmax_last(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x, dim=-1)

    # =========================================================
    # Attention / MLP / Block
    # =========================================================

    def attn(self, X: torch.Tensor, blk: nn.Module) -> torch.Tensor:
        """
        Implements your Attn^(ell)(X):
          [Q,K,V] = X W_qkv^T + b_qkv
          reshape to (B,h,T,d_h)
          S = (Q K^T)/sqrt(d_h)
          causal mask
          A = softmax(S)
          Y = A V
          concat heads -> (B,T,d)
          out = Drop( Y W_o^T + b_o )
        """
        B, T, d = X.shape

        qkv = self.linear(X, blk.W_qkv, blk.b_qkv)          # (B,T,3d)
        Q, K, V = qkv.split(d, dim=-1)                     # each (B,T,d)

        # reshape to (B,h,T,d_h)
        Q = Q.view(B, T, self.cfg.n_head, self.d_h).transpose(1, 2)
        K = K.view(B, T, self.cfg.n_head, self.d_h).transpose(1, 2)
        V = V.view(B, T, self.cfg.n_head, self.d_h).transpose(1, 2)

        # scaled dot-product: (B,h,T,T)
        S = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_h)
        S = self.causal_mask(S, T)
        A = self.softmax_last(S)

        Y = A @ V                                           # (B,h,T,d_h)
        Y = Y.transpose(1, 2).contiguous().view(B, T, d)     # concat heads -> (B,T,d)

        out = self.linear(Y, blk.W_o, blk.b_o)               # (B,T,d)
        out = self.drop(out)
        return out

    def mlp(self, X: torch.Tensor, blk: nn.Module) -> torch.Tensor:
        """
        Z = X W_fc^T + b_fc    -> (B,T,4d)
        G = GELU(Z)
        out = Drop( G W_proj^T + b_proj ) -> (B,T,d)
        """
        Z = self.linear(X, blk.W_fc, blk.b_fc)
        G = self.gelu(Z)
        out = self.linear(G, blk.W_proj, blk.b_proj)
        out = self.drop(out)
        return out

    def block(self, X: torch.Tensor, blk: nn.Module) -> torch.Tensor:
        """
        Pre-LN residual:
          X <- X + Attn( LN1(X) )
          X <- X + MLP ( LN2(X) )
        """
        X = X + self.attn(self.ln(X, blk.gamma_1, blk.beta_1), blk)
        X = X + self.mlp(self.ln(X, blk.gamma_2, blk.beta_2), blk)
        return X

    # =========================================================
    # Full forward (matches your diagram)
    # =========================================================

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        U: (B,T,n_u)
        returns Yhat: (B,T,n_y)
        """
        B, T, n_u = U.shape
        assert n_u == self.cfg.n_u
        assert 1 <= T <= self.cfg.block_size

        # (1) Token embedding: E = U W_e^T + b_e  -> (B,T,d)
        E = self.linear(U, self.W_e, self.b_e)

        # (2) Positional: P = W_p[:T] -> (1,T,d), X0 = Drop(E+P)
        P = self.W_p[:T, :].unsqueeze(0)  # (1,T,d)
        X = self.drop(E + P)

        # (3) Blocks
        for blk in self.blocks:
            X = self.block(X, blk)

        # (4) Final LN
        h = self.ln(X, self.gamma_f, self.beta_f)

        # (5) Head
        Yhat = self.linear(h, self.W_y, self.b_y)
        return Yhat

    # =========================================================
    # Parameter counting (weights + biases)
    # =========================================================

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p is not None)

######################################################################

def ar_rollout(
    batch_u: torch.Tensor,
    model: nn.Module,
    *,
    feedback_channel: int = 4,
) -> torch.Tensor:
    """
    Pure autoregressive rollout (teacher forcing OFF).
    Matches your code logic:
      - clone input and zero feedback channel
      - init yhat_0 = 0
      - for t=0..T-1:
          inject yhat_{t-1} into U[:,t,feedback_channel]
          run model on prefix U[:,:t+1,:]
          read last output as yhat_t
      - stack yhat_1..yhat_T into (B,T,1)
    """
    _assert_shapes(batch_u)

    B, T, n_u = batch_u.shape
    u = batch_u.clone()
    u[:, :, feedback_channel] = 0.0

    # yhat_0 = 0 (no need requires_grad=True; gradients flow through model outputs anyway)
    last_yhat = torch.zeros(B, device=batch_u.device, dtype=batch_u.dtype)

    preds = []

    for t in range(T):
        u_step = u.clone()
        if t > 0:
            u_step[:, t, feedback_channel] = last_yhat  # inject yhat_{t-1}

        prefix = u_step[:, : t + 1, :]                 # (B, t+1, n_u)
        y_full = model(prefix)                          # (B, t+1, n_y)
        last_yhat = y_full[:, -1, 0].contiguous()       # (B,)
        preds.append(last_yhat.unsqueeze(1))            # (B,1)

    yhat = torch.cat(preds, dim=1).unsqueeze(-1)        # (B,T,1)
    return yhat

def mse_objective(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Scalar objective: (1/(B*T*n_y)) * ||yhat - y||_F^2
    """
    _assert_shapes(yhat, y)
    return torch.mean((yhat - y) ** 2)

def batch_objective(
    model: nn.Module,
    batch_u: torch.Tensor,
    batch_y: torch.Tensor,
    *,
    feedback_channel: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes:
      yhat = AR(batch_u; Theta)
      L    = MSE(yhat, batch_y)
    Returns (loss, yhat).
    """
    yhat = ar_rollout(batch_u, model, feedback_channel=feedback_channel)
    loss = mse_objective(yhat, batch_y)
    return loss, yhat

def build_adamw_with_grouping(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    """
    Parameter grouping:
      - decay: tensors with dim >= 2
      - no_decay: tensors with dim < 2 (biases, LN gamma/beta)
    If some biases are disabled via register_parameter(..., None),
    they won't appear in model.parameters() and won’t be optimized.
    """
    decay, no_decay = [], []

    for p in model.parameters():
        if p is None:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)

######################################################################

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_u: torch.Tensor,
    batch_y: torch.Tensor,
    *,
    feedback_channel: int = 4,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    One step of solving the optimisation problem:
      Theta <- Theta - lr * (AdamW direction)
    """
    model.train()
    _assert_shapes(batch_u, batch_y)

    optimizer.zero_grad(set_to_none=True)

    loss, yhat = batch_objective(
        model=model,
        batch_u=batch_u,
        batch_y=batch_y,
        feedback_channel=feedback_channel,
    )

    loss.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

    optimizer.step()

    return {"loss": loss.detach(), "yhat": yhat.detach()}

def fit(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    optim_cfg: OptimConfig,
    device: torch.device,
    epochs: int,
    feedback_channel: int = 4,
) -> None:
    """
    High-level solver for:
      min_Theta E[(1/BT)||AR(U;Theta)-Y||^2]
    using AdamW with fixed lr.
    """
    model.to(device)
    optimizer = build_adamw_with_grouping(model, optim_cfg)

    for ep in range(epochs):
        for batch_u, batch_y in train_loader:
            batch_u = batch_u.to(device)
            batch_y = batch_y.to(device)

            stats = train_step(
                model=model,
                optimizer=optimizer,
                batch_u=batch_u,
                batch_y=batch_y,
                feedback_channel=feedback_channel,
            )

        # optional: print per-epoch
        # print(f"epoch={ep:03d} loss={stats['loss'].item():.6f}")

######################################################################

