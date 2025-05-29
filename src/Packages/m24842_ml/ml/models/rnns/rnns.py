import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from ..common import GatedRMSNorm

class Mamba2Block(nn.Module):
    def __init__(self, d_model, n_layers=24, d_state=128, d_conv=4,
                 expand=2, n_heads=4, chunk_size=64, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
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
            padding=self.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.A_log = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.D = nn.Parameter(torch.empty(self.n_heads, device=device))
        self.norm = GatedRMSNorm(self.d_inner, device=device)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device)
        
        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for stability and training efficiency."""

        # Xavier/Glorot for Linear layers
        nn.init.kaiming_uniform_(self.in_proj.weight, nonlinearity='relu')
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

    def forward(self, u, h=None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        if h:
            return self.step(u, h)

        # Keep track of original sequence length
        seqlen = u.shape[1]
        
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
        
        # Prepare for conv1d
        xBC_conv = xBC.permute(0, 2, 1).contiguous()
        
        # Apply convolution
        conv_out = self.conv1d(xBC_conv)
        
        # Important: Slice to maintain original sequence length
        # The convolution with padding=d_conv-1 produces extra timesteps we don't need
        conv_out = conv_out[:, :, :seqlen]
        
        # Back to [B, L, C]
        xBC = conv_out.permute(0, 2, 1).contiguous()
        xBC = F.silu(xBC)
        
        x, B, C = torch.split(
            xBC, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        
        # Reshape x for the SSM operation
        x = x.reshape(x.shape[0], seqlen, self.n_heads, -1)
        
        dt = F.softplus(dt + self.dt_bias)
        
        y, ssm_state = self.ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            B.unsqueeze(2),
            C.unsqueeze(2),
            self.chunk_size,
            device=self.device,
        )
        
        y = y + x * self.D.unsqueeze(-1)
        y = y.reshape(y.shape[0], seqlen, -1)
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y

    def segsum(self, x, device=None):
        """Stable segment sum calculation.

        `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

        Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
        """
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def ssd(self, x, A, B, C, chunk_size, initial_states=None, device=None):
        """Structed State Space Duality (SSD) - the core of Mamba-2

        This is almost the exact same minimal SSD code from the blog post.

        Arguments
            x: (batch, seqlen, n_heads, d_head)
            A: (batch, seqlen, n_heads)
            B: (batch, seqlen, n_heads, d_state)
            C: (batch, seqlen, n_heads, d_state)

        Return
            y: (batch, seqlen, n_heads, d_head)

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
        L = torch.exp(self.segsum(A, device=device))
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
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
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
                 use_embedding=True, weight_tying=False,
                 bidirectional=False,
                 chunk_size=16, device="cpu"):
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(input_dim, emb_dim, device=device) if use_embedding else nn.Linear(input_dim, emb_dim, bias=False, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer_f=Mamba2Block(d_model=emb_dim, n_layers=n_layers, d_state=emb_dim, d_conv=4, expand=2, n_heads=n_heads, chunk_size=chunk_size, device=device),
                                mixer_b=Mamba2Block(d_model=emb_dim, n_layers=n_layers, d_state=emb_dim, d_conv=4, expand=2, n_heads=n_heads, chunk_size=chunk_size, device=device) if bidirectional else None,
                                norm=GatedRMSNorm(emb_dim, device=device),
                            )
                        )
                        for _ in range(n_layers)
                    ]
                ),
                norm_f=GatedRMSNorm(emb_dim, device=device),
            )
        )
        self.out_proj = nn.Linear(emb_dim, output_dim, bias=False, device=device)
        
        nn.init.xavier_uniform_(self.backbone.embedding.weight)
        if weight_tying: self.out_proj.weight = self.backbone.embedding.weight
        else: nn.init.xavier_uniform_(self.out_proj.weight)
        
        self.to(device)

    def forward(self, x):
        seqlen = x.shape[1]

        x = ((self.input_dim-1)*x).long().squeeze(-1)
        x = self.backbone.embedding(x)
        for i, layer in enumerate(self.backbone.layers):
            y_f = layer.mixer_f(layer.norm(x))
            if self.bidirectional:
                y_b = layer.mixer_b(layer.norm(x.flip(1)))
                x = y_f + y_b + x
            else:
                x = y_f + x

        x = self.backbone.norm_f(x)
        logits = self.out_proj(x)
        return logits[:, :seqlen]

