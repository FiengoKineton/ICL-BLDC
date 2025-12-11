# model.py

"""
Implementation of the Transformer models for dynamical systems. Derived from Karpathy's nanoGPT
https://github.com/karpathy/nanoGPT/
"""

import math, inspect, torch, torch.nn as nn
from dataclasses import dataclass
from transformer_utils import LayerNorm, Block


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
    activation_function: str = "gelu"  # "gelu", "relu", etc.


class GPT(nn.Module):

    def __init__(self, config, print_flag: bool = True):
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
        if print_flag: print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, print_flag: bool = True):
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
        if print_flag: print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        if print_flag: print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        if print_flag: print(f"using fused AdamW: {use_fused}")

        return optimizer

