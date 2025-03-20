import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract
import math
from typing import Optional
import time
import matplotlib.pyplot as plt

class AssociativeMemoryBlock(nn.Module):
    """
    An associative memory block that performs multi-step retrieval and updates an internal memory.
    
    Args:
        embedding_dim (int): Dimension of the input embeddings.
        memory_dim (int): Dimension for memory representations (used for V, Q, and K).
        retrieval_rate (int): Rate at which memory is retrieved. Default is 2.
        retrieval_depth (int): Depth of multi-step retrieval. Default is 1.
        n_heads (int): Number of attention heads. Default is 1.
        chunk_size (Optional[int]): Optional chunk size (currently unused).
        device (Optional[torch.device]): Device on which to run the module.
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        memory_dim: int, 
        retrieval_rate: int = 2, 
        retrieval_depth: int = 1, 
        chunk_size: Optional[int] = None, 
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.v_dim = memory_dim
        self.q_k_dim = memory_dim
        self.retrieval_rate = retrieval_rate
        self.retrieval_depth = retrieval_depth
        self.chunk_size = chunk_size
        
        self.V = nn.Linear(self.embedding_dim, self.v_dim, bias=False)
        self.Q = nn.Parameter(torch.empty((self.retrieval_rate, self.v_dim, self.q_k_dim)), requires_grad=True)
        self.K = nn.Linear(self.embedding_dim, self.q_k_dim, bias=False)
        self.out_proj = nn.Linear(self.v_dim, self.embedding_dim, bias=False)
        
        self.init_weights()
        
        self.device = device if device else torch.device('cpu')
        self.to(self.device)
        
    def init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def calc_retrieval_index(self, itr: int) -> int:
        """Calculate the retrieval index for a given iteration."""
        
        if self.retrieval_rate == 1: return itr
        return int((self.retrieval_rate**itr - 1) / (self.retrieval_rate - 1))
        
    def forward(self, x: torch.Tensor, M: Optional[torch.Tensor] = None):
        """
        Forward pass of the AssociativeMemoryBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).
            M (Optional[torch.Tensor]): Memory tensor of shape (batch_size, seq_len, v_dim, q_k_dim).
                                        If not provided, it is initialized to zeros.
            return_qkv (bool): If True, returns (Q_retrieval, K_retrieval, V_retrieval, M_final).
        
        Returns:
            Tuple containing:
                - output (if return_qkv is False): The projected retrieval V tensor.
                - M_final: The final memory state of shape (batch_size, v_dim, q_k_dim).
            Or, if return_qkv is True:
                (Q_retrieval, K_retrieval, V_retrieval, M_final)
        """
        
        batch_size, seq_len, _ = x.size()
        
        if M is None: M = torch.zeros((batch_size, self.v_dim, self.q_k_dim), device=self.device)
        
        V = self.V(x)
        K = F.normalize(self.K(x), p=2, dim=-1)
        
        VK = contract('bsv, bsk -> bsvk', V, K, optimize='auto')
        KK = contract('bsk, bsK -> bskK', K, K, optimize='auto')
        M_all = torch.empty((batch_size, seq_len, self.v_dim, self.q_k_dim), device=self.device)
        
        for i in range(seq_len):
            M = M - contract('bvk, bkK -> bvK', M, KK[:, i], optimize='auto') + VK[:, i]
            M_all[:, i].copy_(M)
        
        # plt.figure(figsize=(8, 8))
        # plt.subplot(211)
        # idx = -2
        # plt.plot(V[0, :, idx].detach().cpu().numpy(), label='V_orig')
        # plt.plot(contract('bsvk, bsk -> bsv', M_all, K)[0, :, idx].detach().cpu().numpy(), label='V_recon_all')
        # plt.plot(contract('bvk, bsk -> bsv', M_all[:, -1], K)[0, :, idx].detach().cpu().numpy(), label='V_recon_last')
        # plt.legend()
        
        # plt.subplot(212)
        # plt.plot((V - contract('bsvk, bsk -> bsv', M_all, K))[0].abs().sum(-1).detach().cpu().numpy(), label='V_recon_err_all')
        # plt.plot((V - contract('bvk, bsk -> bsv', M_all[:, -1], K))[0].abs().sum(-1).detach().cpu().numpy(), label='V_recon_err_last')
        # plt.ylim(-10, None)
        # plt.legend()
        # plt.show()
        
        # === Memory Retrieval ===
        retrieval_count = self.calc_retrieval_index(self.retrieval_depth+1)
        
        V_retrieval = torch.zeros((batch_size, seq_len, retrieval_count, self.v_dim), device=self.device)
        V_retrieval[:, :, 0] = V
        
        Q_retrieval = contract('bsv, rvq -> bsrq', V, self.Q, optimize='auto')
        Q_retrieval = F.normalize(Q_retrieval, p=2, dim=-1)
        for i in range(1, self.retrieval_depth+1):
            idx_0 = self.calc_retrieval_index(i)
            idx_1 = self.calc_retrieval_index(i+1)
            
            V_retrieval[:, :, idx_0:idx_1] = contract('bsvk, bsrk -> bsrv', M_all, Q_retrieval, optimize='auto') # (batch_size, retrieval_count, v_dim)
            V_retrieval_reshaped = V_retrieval[:, :, idx_0:idx_1].reshape(batch_size, seq_len, -1, self.v_dim) # (batch_size, seq_len, pre_itr_retrieval_count / retrieval_rate, retrieval_rate, v_dim)
            
            Q_retrieval = contract('bsnv, rvq -> bsnrq', V_retrieval_reshaped, self.Q, optimize='auto').reshape(batch_size, seq_len, -1, self.q_k_dim) # (batch_size, seq_len, curr_itr_retrieval_count, q_k_dim)
            Q_retrieval = F.normalize(Q_retrieval, p=2, dim=-1)

        M_final = M_all[:, -1] # (batch_size, v_dim, q_k_dim)
        
        return self.out_proj(V_retrieval), M_final
    
class AssociativeAttentionBlock(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        memory_dim: int, 
        retrieval_rate: int = 2, 
        retrieval_depth: int = 1, 
        n_heads: int = 1, 
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.retrieval_rate = retrieval_rate
        self.retrieval_depth = retrieval_depth
        self.n_heads = n_heads

        self.in_norm = nn.LayerNorm(embedding_dim)
        self.memory = AssociativeMemoryBlock(embedding_dim, memory_dim, retrieval_rate, retrieval_depth, device=device)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(embedding_dim)
        
        self.device = device if device else torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, _ = x.size()
        x = self.in_norm(x)
        x, M = self.memory(x)
        x = rearrange(x, 'b s r d -> (b s) r d')
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = rearrange(x, '(b s) r d -> b s r d', b=batch_size, s=seq_len)
        x = self.out_norm(x)
        return x, M