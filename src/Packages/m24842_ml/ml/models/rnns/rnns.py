import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..common import *

cuda_causal_conv1d = None
try: from causal_conv1d import causal_conv1d_fn as cuda_causal_conv1d
except: pass

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4,
                 expand=2, n_heads=4, chunk_size=64, device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size
        self.d_inner = expand * d_model
        self.n_heads = n_heads
        self.device = device

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, device=device)

        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.d_conv,
            groups=conv_dim,
            padding=self.d_conv-1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.A_log = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.D = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.norm = GatedRMSNorm(self.d_inner, device=device)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier/Glorot for Linear layers
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Kaiming He Initialization for Conv1D
        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize dt_bias (time-step scaling factor)
        nn.init.uniform_(self.dt_bias, -0.5, 0.5)

        # Initialize A_log with small negative values for stability
        nn.init.uniform_(self.A_log, -0.1, -0.01)

        # Initialize D with small values (identity-like for residual connections)
        nn.init.uniform_(self.D, 0.9, 1.1)

    def forward(self, u):
        """
        Arguments
            u: (batch, seq_len, d_model) input. seq_len should be a multiple of chunk_size.

        Return (y, h)
            y: (batch, seq_len, d_model) output
        """
        # Keep track of original sequence length
        seq_len = u.shape[1]
        u = F.pad(u, (0, 0, 0, self.chunk_size - seq_len % self.chunk_size), value=0.0)
        padded_seq_len = u.shape[1]
        
        A = -torch.exp(self.A_log)
        zxbcdt = self.in_proj(u)
        
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.d_state,
                self.n_heads,
            ],
            dim=-1,
        )
        
        # Apply convolution
        if cuda_causal_conv1d is not None:
            xBC = cuda_causal_conv1d(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            ).transpose(1, 2)
        else:
            xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :padded_seq_len])
        
        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.d_state,
                self.d_state,
            ],
            dim=-1
        )
        
        # Reshape x for the SSM operation
        x = x.reshape(x.shape[0], padded_seq_len, self.n_heads, -1)
        
        dt = F.softplus(dt + self.dt_bias)
        
        y, ssm_state = self.ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            B.unsqueeze(2),
            C.unsqueeze(2),
            self.chunk_size,
        )
        
        y = y + x * self.D.unsqueeze(-1)
        y = y.reshape(y.shape[0], padded_seq_len, -1)
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y[:, :seq_len]

    def segsum(self, x):
        """Stable segment sum calculation.

        `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

        Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
        """
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=self.device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C, chunk_size, initial_states=None):
        """Structed State Space Duality (SSD) - the core of Mamba-2

        This is almost the exact same minimal SSD code from the blog post.

        Arguments
            x: (batch, seq_len, n_heads, d_head)
            A: (batch, seq_len, n_heads)
            B: (batch, seq_len, n_heads, d_state)
            C: (batch, seq_len, n_heads, d_state)

        Return
            y: (batch, seq_len, n_heads, d_head)

        Source
        1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
        2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
        """
        assert x.shape[1] % chunk_size == 0

        # Rearrange into chunks
        # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
        # This is not implemented and left as an exercise for the reader 😜
        x, A, B, C = [
            rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
        ]

        A = rearrange(A, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        L = torch.exp(self.segsum(A))
        Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

        return Y, final_state

class Mamba2(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1,
                 d_state=None, d_conv=4, expand=2,
                 use_embedding=True, weight_tying=False,
                 bidirectional=False,
                 chunk_size=64, device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_state = d_state if d_state is not None else emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        self.use_embedding = use_embedding
        self.device = device

        if use_embedding:
            self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else:
            self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    mixer=Mamba2Block(
                        d_model=emb_dim,
                        d_state=self.d_state,
                        d_conv=d_conv,
                        expand=expand,
                        n_heads=n_heads,
                        chunk_size=chunk_size,
                        device=device
                    ),
                    norm=GatedRMSNorm(emb_dim, device=device),
                )
            ) for _ in range(n_layers)
        ])
        self.norm_f=GatedRMSNorm(emb_dim, device=device)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)

    def forward(self, x):
        seq_len = x.shape[1]
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x_norm = layer.norm(x)
            y_f = layer.mixer(x_norm)
            y_b = layer.mixer(x_norm.flip(1)) if self.bidirectional else 0.0
            x = x + y_f + y_b
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x[:, :seq_len]

class SeqLinear(nn.Module):
    """
    Linear layer with weights derived from a weighted sum of rank-1 sequential updates.
    """
    def __init__(self, d_model, n_heads, bias=True,
                 W_0=False, d_conv=4, split_zu=False,
                 zu_dim=None, v_dim=None,
                 device="cpu"):
        super().__init__()
        self.d_model = d_model
        self.zu_dim = zu_dim if zu_dim is not None else d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.n_heads = n_heads
        self.split_zu = split_zu
        self.bias = bias
        self.device = device
        
        d_in_proj = 2 * self.zu_dim + self.v_dim + self.n_heads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, device=device)
        self.dt_bias = nn.Parameter(torch.empty(n_heads, device=device))
        self.conv = nn.Conv1d(
            in_channels=d_in_proj,
            out_channels=d_in_proj,
            kernel_size=d_conv,
            groups=d_in_proj,
            padding=d_conv-1,
            device=device,
        )
        self.W_0 = nn.Parameter(torch.empty(n_heads, self.zu_dim, self.v_dim // n_heads, device=device)) if W_0 else None
        self.out_proj = nn.Linear(self.v_dim, d_model, bias=False, device=device)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.uniform_(self.dt_bias, -0.5, 0.5)
        
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        
        if self.W_0 is not None: nn.init.constant_(self.W_0, 0.)
    
    def forward(self, x, W_0=None):
        bsz, src_len, d_model = x.size()
        
        x = self.in_proj(x)
        
        if cuda_causal_conv1d is not None:
            x = cuda_causal_conv1d(
                x.transpose(1, 2),
                rearrange(self.conv.weight, "d 1 w -> d w"),
                bias=self.conv.bias,
                activation="silu",
            ).transpose(1, 2)
        else:
            x = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2)[:, :src_len])
        
        z, u, v, dt = torch.split(
            x,
            [
                self.zu_dim,
                self.zu_dim,
                self.v_dim,
                self.n_heads,
            ],
            dim=-1
        )
        
        dt = F.softplus(dt + self.dt_bias)
        if self.split_zu:
            z = rearrange(z, 'b s (h d) -> b s h d', h=self.n_heads).contiguous()
            u = rearrange(u, 'b s (h d) -> b s h d', h=self.n_heads).contiguous()
        else:
            z = z.unsqueeze(-2)
            u = u.unsqueeze(-2)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_heads).contiguous()
        
        W = torch.cumsum(torch.einsum('bsh, bshd, bshe -> bshde', dt, u, v), dim=1)
        if W_0 is not None: W = W + W_0.unsqueeze(1)
        elif self.W_0 is not None: W = W + self.W_0.unsqueeze(1)
        
        norm = torch.cumsum(dt, dim=1).unsqueeze(-1)
        
        out = torch.matmul(z.unsqueeze(-2), W).squeeze(-2) / norm
        out = rearrange(out, 'b s h d -> b s (h d)')
        out = self.out_proj(out)
        return out

class SeqMLP(nn.Module):
    def __init__(self, emb_dim, input_dim, output_dim,
                 n_layers=1, n_heads=1, dropout=0.0,
                 zu_dim=None, v_dim=None, W_0=False,
                 d_conv=4,
                 use_embedding=True, weight_tying=False,
                 bidirectional=True,
                 bias=True, device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        self.use_embedding = use_embedding
        self.device = device

        if use_embedding:
            self.embedding = nn.Embedding(input_dim, emb_dim, device=device)
        else:
            self.embedding = nn.Linear(input_dim, emb_dim, bias=False, device=device)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict(
                dict(
                    mixer=SeqLinear(
                        d_model=emb_dim,
                        n_heads=n_heads,
                        zu_dim=zu_dim,
                        v_dim=v_dim,
                        W_0=W_0,
                        d_conv=d_conv,
                        bias=bias,
                        device=device
                    ),
                    dropout = nn.Dropout(dropout),
                    norm=nn.RMSNorm(emb_dim, device=device),
                )
            ) for _ in range(n_layers)
        ])
        self.norm_f=nn.RMSNorm(emb_dim, device=device)
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        nn.init.xavier_uniform_(self.embedding.weight)
        if weight_tying: self.out_proj.weight = self.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)

    def forward(self, x):
        seq_len = x.size(1)
        if self.use_embedding: x = self.embedding(x.long())
        else: x = self.embedding(x)
        for layer in self.layers:
            x_norm = layer.norm(x)
            y_f = layer.dropout(layer.mixer(x_norm))
            y_b = layer.dropout(layer.mixer(x_norm.flip(1))) if self.bidirectional else 0.0
            x = x + y_f + y_b
        x = self.norm_f(x)
        x = self.out_proj(x)
        return x