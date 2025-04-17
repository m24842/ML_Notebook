import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract
import math
from typing import Optional

class FAVORPlusKernel(nn.Module):
    def __init__(self, d_model, num_features=256, ortho_random_features=True):
        """
        d_model: Hidden dimension size (key/query dimension)
        num_features: Number of random features for approximation
        ortho_random_features: If True, uses orthogonal random features (improves accuracy)
        """
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        
        # Initialize random feature matrix for kernel approximation
        self.random_matrix = self._create_random_features(d_model, num_features, ortho_random_features)
    
    def _create_random_features(self, dim, num_features, ortho):
        """
        Generates the random feature matrix for FAVOR+ approximation.
        If `ortho=True`, uses orthogonal random features for better accuracy.
        """
        W = torch.randn(dim, num_features)
        if ortho:
            W, _ = torch.linalg.qr(W)  # Make it orthogonal
        return W  # Size: (d_model, num_features)

    def forward(self, x):
        """
        Applies the FAVOR+ kernel feature map approximation.
        φ(x) = exp(Wx - ||x||^2 / 2)
        """
        x_proj = torch.einsum("bnd,dm->bnm", x, self.random_matrix)  # (batch, n, num_features)
        return torch.exp(x_proj - torch.norm(x, dim=-1, keepdim=True) ** 2 / 2)

class AssociativeMemoryBlock(nn.Module):
    """
    An associative memory block that performs multi-step retrieval and updates an internal memory.
    
    Args:
        d_model (int): Dimension of the input.
        d_state (int): Dimension for memory state (used for V, Q, and K).
        n_heads (int): Number of attention heads. Default is 1.
        chunk_size (Optional[int]): Optional chunk size (currently unused).
        device (Optional[torch.device]): Device on which to run the module.
    """
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int = 1, 
        chunk_size: Optional[int] = None, 
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.chunk_size = chunk_size
        
        self.w_0 = nn.Parameter(torch.ones(nhead))
        self.M_0 = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        self.M_norm_0 = nn.Parameter(torch.ones(self.d_model, self.d_model))
        self.alpha = nn.Linear(self.d_model, nhead, bias=False)
        self.Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.K = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self._init_weights()
        
        self.device = device if device else torch.device('cpu')
        self.to(self.device)
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        
        nn.init.xavier_uniform_(self.alpha.weight)
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.K.weight)
    
    def _state_pscan(self, a, w, Q, chunk_size: Optional[int] = None):
        """
        Perform parallel scan on matrix valued elements of the form D + outer(Q, Q).

        Args:
            a (torch.tensor(batch_size, seq_len)): Memory gate.
            w (torch.tensor(batch_size, seq_len)): Memory decay.
            Q (torch.tensor(batch_size, seq_len, d_model)): Queries.
            chunk_size (Optional[int]): Override initialized chunk size.
        """
        
        batch_size, seq_len, _ = Q.size()
        if chunk_size is None: chunk_size = self.chunk_size
        
        num_chunks = math.ceil(seq_len / chunk_size)
        chunk_padded_seq_len = chunk_size * num_chunks
        
        D = (1 - a) * (1 - w)
        
        D_padded = F.pad(D, (0, 0, chunk_padded_seq_len - seq_len, 0), value=1.0)
        Q_padded = F.pad(Q, (0, 0, chunk_padded_seq_len - seq_len, 0), value=0.0)
        w_padded = F.pad(w, (0, 0, chunk_padded_seq_len - seq_len, 0), value=0.0)
        
        D_expanded = D_padded.unsqueeze(1).repeat(1, seq_len, 1, 1)
        Q_expanded = Q_padded.unsqueeze(1).repeat(1, seq_len, 1, 1)
        w_expanded = w_padded.unsqueeze(1).repeat(1, seq_len, 1, 1)
                
        D_chunked = D_expanded.reshape(batch_size, seq_len, num_chunks, chunk_size, self.nhead)
        Q_chunked = Q_expanded.reshape(batch_size, seq_len, num_chunks, chunk_size, self.d_model)
        w_chunked = w_expanded.reshape(batch_size, seq_len, num_chunks, chunk_size, self.nhead)
        
        I = torch.eye(self.d_model, device=self.device).reshape(1, 1, 1, self.d_model, self.d_model).repeat(batch_size, seq_len, num_chunks, 1, 1)
        seq_prod = D_chunked[:, :, :, 0].unsqueeze(-1) * I + torch.einsum('...w, ...D, ...d -> ...Dd', w_chunked[:, :, :, 0], Q_chunked[:, :, :, 0], Q_chunked[:, :, :, 0])
        for i in range(1, chunk_size):
            seq_prod = torch.einsum('...Dd, ...g -> ...Dd', seq_prod, D_chunked[:, :, :, i]) + torch.einsum('...Dd, ...w, ...D, ...d -> ...Dd', seq_prod, w_chunked[:, :, :, i], Q_chunked[:, :, :, i], Q_chunked[:, :, :, i])
        
        scan_count = math.ceil(math.log2(num_chunks + 1))
        padded_scan_len = 2 ** scan_count
        
        output = F.pad(seq_prod, (0, 0, 0, 0, 0, padded_scan_len - num_chunks), value=0.0)
        output[:, :, :padded_scan_len - num_chunks] = torch.eye(self.d_model).reshape(1, 1, self.d_model, self.d_model).repeat(1, padded_scan_len - num_chunks, 1, 1)
        
        for i in range(scan_count):
            output = output.reshape(batch_size, seq_len, -1, 2, self.d_model, self.d_model)
            output = torch.einsum('...Ab, ...bC -> ...AC', output[:, :, :, 0], output[:, :, :, 1])
        
        output = output.reshape(batch_size, seq_len, self.d_model, self.d_model)
        
        return output
        
    def forward(self, x, M=None, M_norm=None):
        """
        Parallel mode sequence processing.

        Args:
            x (torch.tensor(batch_size, seq_len, d_model)): Input sequence.
            M (torch.tensor(d_model, d_model), optional): Optional initial memory state.

        Returns:
            output: Attention output.
        """
        batch_size, seq_len, _ = x.size()
        
        t = torch.arange(2, seq_len+2, dtype=torch.long, device=self.device)
        w = 1 / (self.w_0 + t + 2)
        w = w.reshape(1, seq_len, self.nhead)
        
        a, q, k, v = self.alpha(x), self.Q(x), self.K(x), self.V(x)
        a = F.sigmoid(a)
        q_exp, k_exp = torch.exp(q), torch.exp(k)
        
        if M is None:
            M = self.M_0.reshape(1, 1, self.d_model, self.d_model).repeat(batch_size, 1, 1, 1)
            M_norm = self.M_norm_0.reshape(1, 1, self.d_model, self.d_model).repeat(batch_size, 1, 1, 1)
        else:
            M = M.unsqueeze(1)
            M_norm = M_norm.unsqueeze(1)
        
        vk = torch.einsum('...sn, ...sD, ...sd -> ...sDd', w * a, v, k_exp)
        vk = torch.cat((M, vk), dim=1)
        
        norm = torch.einsum('...sn, ...sD, ...sd -> ...sDd', w * a, torch.ones((1, 1, self.d_model), device=self.device), k_exp)
        M_norm = torch.cat((M_norm, norm), dim=1)
        
        a = torch.cat((torch.ones((batch_size, 1, 1), device=self.device), a), dim=1)
        w = torch.cat((self.w_0.reshape(1, 1, self.nhead).repeat(batch_size, 1, 1), w), dim=1)
        q_exp = torch.cat((torch.zeros_like(q_exp[:, 0:1]), q_exp), dim=1)
        
        state_decay = self._state_pscan(a, w, q_exp)
        
        M = torch.einsum('...Ab, ...bC -> ...AC', vk, state_decay)
        M_norm = torch.einsum('...Ab, ...bC -> ...AC', M_norm, state_decay)
        
        output = torch.einsum('...Dd, ...d -> ...D', M, q_exp)
        output = output / torch.einsum('...Dd, ...d -> ...D', M_norm, q_exp)
        
        return output

device = torch.device('mps')
d_model = 5
seq_len = 30
test = AssociativeMemoryBlock(d_model, 1, 10, device)

x = torch.rand(1, seq_len, d_model, device=device)
out = test(x)
print(x.shape)