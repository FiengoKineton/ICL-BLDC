"""
Implementation of the Transformer models for dynamical systems. Derived from Karpathy's nanoGPT
https://github.com/karpathy/nanoGPT/
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Zero-step Transformer for dynamical sequences (ICL-style estimator).
# Input:  batch_u ∈ ℝ^{B×T×n_u}  (continuous tokens per time step, e.g. [vα,vβ,iα,iβ, ω̂_{k-1}])
# Output: batch_y_pred ∈ ℝ^{B×T×n_y} (per-step predictions, e.g. ω̂_{1:T}); at inference usually take x[:, -1, :].
# Causality: strictly causal self-attention (no peeking ahead).
# Math (per head): α_{tj} = softmax((q_t k_j^T)/√d_h) for j ≤ t; y_t = Σ_j α_{tj} v_j.
# Related works:
#  - ICL state estimators: Busetto et al. (IFAC 2024); speed ICL on BLDC: Colombo et al. (2025)
#  - Decoder-only control/estimation with context windows: "One controller to rule them all" (2025)
# -----------------------------------------------------------------------------


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
#def new_gelu(x):
#    """
#    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
#    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
#    """
#    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (to mimic GPT-2 behavior). PyTorch doesn't support simply bias=False
    Args:
        ndim: embedding width d
        bias: if False, drop β parameter; PyTorch's LN doesn't expose this flag natively.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x: (B,T,d). Project to q,k,v then split heads -> (B,h,T,d_h)
        # Attention math (per head):
        # att = softmax((Q K^T) / sqrt(d_h)) with future positions masked
        # y = att @ V  -> (B,h,T,d_h), then concat heads -> (B,T,d)
        # Dropout applied on att (attn_dropout) and residual output (resid_dropout).

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network:
       x -> Linear(d,4d) -> GELU -> Linear(4d,d) -> Dropout.
       This is the non-linear mixing after attention in each Transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        #x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer decoder block (Pre-LN, residual):
       x = x + Attn(LN(x))
       x = x + MLP(LN(x))
       Pre-normalization eases optimization in deep stacks.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    # block_size: max sequence length T (also size of positional embedding table)
    # n_layer:   number of decoder blocks
    # n_head:    number of attention heads
    # n_embd:    embedding/hidden width d
    # n_x/n_u/n_y: dims for states/inputs/outputs (here we use n_u at input, n_y at output)
    # dropout:   dropout prob for attention + MLP residuals
    # bias:      include bias in Linear/LayerNorm (GPT-2 style if True)

    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_x: int = 2
    n_u: int = 1
    n_y: int = 1
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(config.n_u, config.n_embd),  # we process continuous data
            #wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_y, bias=True) # False
        #self.lm_head = nn.Linear(config.n_embd, config.n_y, bias=False) # False

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch_u):
        # batch_u: (B,T,n_u). T must be ≤ block_size (positional table length).
        # pos: (1,T) integer indices -> positional embeddings P ∈ ℝ^{T×d}
        # Tokenization for continuous streams:
        #   E_t = W_e u_t  (Linear), X_t = E_t + P_t
        #   H   = Transformer(X)  (causal; each H_t sees X_≤t only)
        # Output head:
        #   Y = lm_head(H) ∈ ℝ^{B×T×n_y}
        # In training: you can supervise all steps (seq2seq regression)
        # In inference: usually take the last step prediction:
        #   y_T = Y[:, -1, :]   # uncomment if you need only current-step output

        device = batch_u.device
        b, t, nu = batch_u.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(batch_u)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # if we are given some desired targets also calculate the loss
        x = self.lm_head(x)

        # Select the last element of the sequence
        # x = x[:, -1, :]  # shape (b, n_embd)
        batch_y_pred = x

        # NOTE: if you always want only the last time-step prediction, replace:
        #   batch_y_pred = x
        # with:
        #   batch_y_pred = x[:, -1, :].unsqueeze(1)  # keep time dim if needed

        return batch_y_pred

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


def warmup_cosine_lr(iter, lr, min_lr, warmup_iters, lr_decay_iters):
    # LR schedule:
    # 1) Warmup: lr * (iter / warmup_iters), for iter < warmup_iters
    # 2) Cosine decay from lr → min_lr over [warmup_iters, lr_decay_iters]
    #    lr(iter) = min_lr + 0.5*(lr - min_lr)*(1 + cos(pi * progress))
    # 3) Clamp to min_lr after lr_decay_iters

    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return lr * iter / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)



# GOTCHA: If T > config.block_size you’ll crash on wpe lookup; either increase block_size
#          or chunk your context window.

# GOTCHA: Flash attention path assumes PyTorch ≥ 2.0. Fallback mask allocates (block_size×block_size);
#          keep block_size reasonable on small GPUs.

# GOTCHA (inference): By default the model returns per-step outputs (B,T,n_y).
#          If your downstream loop expects only the current estimate, select x[:, -1, :].
