import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, activation=F.relu, bias=True, device="cpu"):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_in
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias, device=device)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias, device=device)
        self.activation = activation
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class GatedRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5, device=None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * F.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class DilatedSlidingWindow(Function):
    @staticmethod
    def forward(ctx, x, size, stride, dilation, dim, pad):
        if torch.is_grad_enabled() and x.requires_grad:
            ctx.size = size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.dim = dim
            ctx.pad = pad
            ctx.x_shape = x.shape
            ctx.save_for_backward(x)
        
        ndim = x.dim()
        if dim < -ndim or dim >= ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [-{ndim}, {ndim-1}], but got {dim})")
        
        if dim < 0:
            dim = ndim + dim

        if pad[0] > 0 or pad[1] > 0:
            pad_tuple = [0] * (2 * ndim)
            pad_idx = 2 * (ndim - 1 - dim)
            pad_tuple[pad_idx] = pad[0]
            pad_tuple[pad_idx + 1] = pad[1]
            x = F.pad(x, tuple(pad_tuple))

        n_padded = x.shape[dim]
        effective_window_size = (size - 1) * dilation + 1
        num_windows = (n_padded - effective_window_size) // stride + 1

        if num_windows <= 0:
            final_shape = list(x.shape)
            final_shape[dim:dim+1] = [0, size]
            return torch.empty(final_shape, dtype=x.dtype, device=x.device)

        out_shape = list(x.shape)
        out_shape[dim:dim+1] = [num_windows, size]

        original_strides = x.stride()
        element_stride = original_strides[dim]

        out_stride = (
            original_strides[:dim] 
            + (stride * element_stride, dilation * element_stride) 
            + original_strides[dim+1:]
        )
        y = x.as_strided(out_shape, tuple(out_stride))

        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not ctx.saved_tensors:
            return None, None, None, None, None, None
        
        x, = ctx.saved_tensors  # This is now the original input
        
        # Create grad_input for the original input shape
        grad_input = torch.zeros_like(x)
        
        # Apply padding to grad_input if needed (to match forward pass)
        ndim = x.dim()
        dim = ctx.dim
        if dim < 0:
            dim = ndim + dim
            
        if ctx.pad[0] > 0 or ctx.pad[1] > 0:
            pad_tuple = [0] * (2 * ndim)
            pad_idx = 2 * (ndim - 1 - dim)
            pad_tuple[pad_idx] = ctx.pad[0]
            pad_tuple[pad_idx + 1] = ctx.pad[1]
            grad_input_padded = F.pad(grad_input, tuple(pad_tuple))
        else:
            grad_input_padded = grad_input

        n_padded = grad_input_padded.shape[dim]
        effective_window_size = (ctx.size - 1) * ctx.dilation + 1
        num_windows = (n_padded - effective_window_size) // ctx.stride + 1

        if num_windows > 0:
            out_shape = list(grad_input_padded.shape)
            out_shape[dim:dim+1] = [num_windows, ctx.size]
            original_strides = grad_input_padded.stride()
            element_stride = original_strides[dim]
            out_stride = (
                original_strides[:dim] 
                + (ctx.stride * element_stride, ctx.dilation * element_stride) 
                + original_strides[dim+1:]
            )
            grad_input_padded.as_strided(out_shape, out_stride).add_(grad_output)

        # If we padded, we need to "unpad" to get back to original input size
        if ctx.pad[0] > 0 or ctx.pad[1] > 0:
            # Extract the original region from the padded gradient
            slices = [slice(None)] * ndim
            slices[dim] = slice(ctx.pad[0], grad_input_padded.shape[dim] - ctx.pad[1] if ctx.pad[1] > 0 else None)
            grad_input = grad_input_padded[tuple(slices)]
        else:
            grad_input = grad_input_padded
        
        return grad_input, None, None, None, None, None

def dilated_sliding_window(x, size, stride=1, dilation=1, dim=-1, pad=(0, 0)):
    return DilatedSlidingWindow.apply(x, size, stride, dilation, dim, pad)

class DilatedSlidingWindowNoPad(Function):
    def _build_index(num_windows, size, stride, dilation, pad_left, N, device):
        # pad_left in elements
        # Compute "center offset" L in dilation-steps
        L = pad_left // dilation
        # window offsets relative to each window start
        offsets = (torch.arange(size, device=device) - L) * dilation  # (size,)
        # base start positions for each window
        base = torch.arange(num_windows, device=device) * stride
        # full index map: (num_windows, size)
        idx = base.unsqueeze(1) + offsets.unsqueeze(0)
        return idx
    
    @staticmethod
    def forward(ctx, x, size, stride, dilation, dim, pad):
        # Only stride>=1 supported
        if torch.is_grad_enabled() and x.requires_grad:
            ctx.size = size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.dim = dim
            ctx.pad = pad
            ctx.x_shape = x.shape
            ctx.save_for_backward(x)

        # bring [batch, seq, ...] ordering
        if dim < 0:
            dim = x.dim() + dim
        # permute to (B, N, D_flat)
        orig_dims = list(range(x.dim()))
        perm = [d for d in orig_dims if d not in (0, dim)]
        perm = [0, dim] + perm
        x_perm = x.permute(perm)
        B, N = x_perm.shape[:2]
        D_flat = torch.tensor(x_perm.shape[2:]).prod()
        x_flat = x_perm.reshape(B, N, D_flat)

        # compute padded length and number of windows
        pad_left, pad_right = pad
        eff = (size - 1) * dilation + 1
        n_padded = N + pad_left + pad_right
        num_windows = (n_padded - eff) // stride + 1
        if num_windows <= 0:
            return x.new_empty((B, 0, size) + tuple(x_perm.shape[2:]))

        # build indices
        idx = DilatedSlidingWindowNoPad._build_index(num_windows, size, stride, dilation, pad_left, N, x.device)
        # mask out-of-bounds
        mask = (idx < 0) | (idx >= N)
        idx_clamped = idx.clamp(0, N-1)

        # gather windows: (B, num_windows, size, D_flat)
        idx_exp = idx_clamped.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D_flat)
        windows = x_flat.unsqueeze(1).expand(-1, num_windows, -1, -1)
        windows = windows.gather(2, idx_exp)
        # zero pad positions
        windows = windows.masked_fill(mask.unsqueeze(0).unsqueeze(-1), 0)

        # reshape feature dims
        out = windows.view((B, num_windows, size) + tuple(x_perm.shape[2:]))

        # permute back to original ordering: insert window dim after batch
        # original perm rearranged dims: [0, dim, ...]
        # we want [batch, num_windows, window_size, ...]
        inv = []
        # build inverse of perm: perm maps new->old; we want to map old->new
        for i,p in enumerate(perm): inv.append(i)
        # now inv gives for each old-axis its index in perm
        # new_tensor dims are [0(batch),1:num_windows,2:size, then features]
        # features are at positions 3...; they correspond to orig dims except 0,dim
        # so output order = [0, 1, 2] + [inv[d] + (0 if inv[d]<2 else 1) for d in orig_dims if d not in (0,dim)]
        feats = [inv[d] + 1 for d in orig_dims if d not in (0, dim)]
        order = [0, 1, 2] + feats
        out = out.permute(order)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        if not ctx.saved_tensors:
            return None, None, None, None, None, None
        
        x, = ctx.saved_tensors
        size = ctx.size
        stride = ctx.stride
        dilation = ctx.dilation
        dim = ctx.dim
        pad_left, pad_right = ctx.pad

        # permute grad_out same as forward pre-permute
        if dim < 0:
            dim = x.dim() + dim
        orig_dims = list(range(x.dim()))
        perm = [d for d in orig_dims if d not in (0, dim)]
        perm = [0, dim] + perm
        # grad_out shape: (B, num_windows, size, *features)
        grad_perm = grad_out.permute([0,1,2] + [i+3 for i in range(grad_out.dim()-3)])
        B, num_windows, S = grad_perm.shape[:3]
        D_flat = torch.tensor(grad_perm.shape[3:]).prod()
        g_flat = grad_perm.reshape(B, num_windows, S, D_flat)

        # build index and mask
        N = x.size(dim)
        idx = DilatedSlidingWindowNoPad._build_index(num_windows, size, stride, dilation, pad_left, N, x.device)
        mask = (idx < 0) | (idx >= N)
        idx_clamped = idx.clamp(0, N-1)

        # mask grads
        g_flat = g_flat.masked_fill(mask.unsqueeze(0).unsqueeze(-1), 0)

        # scatter-add
        grad_flat = torch.zeros((B, N, D_flat), device=x.device, dtype=x.dtype)
        idx_exp = idx_clamped.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, D_flat)
        grad_flat.scatter_add_(1,
            idx_exp.reshape(B, num_windows * S, D_flat),
            g_flat.reshape(B, num_windows * S, D_flat)
        )

        # reshape back
        grad = grad_flat.view((B, N) + tuple(x.permute(perm).shape[2:]))
        # inverse perm
        inv = [0]*len(perm)
        for i,p in enumerate(perm): inv[p] = i
        grad = grad.permute(inv)
        return grad, None, None, None, None, None

def dilated_sliding_window_nopad(x, size, stride=1, dilation=1, dim=-1, pad=(0,0)):
    return DilatedSlidingWindowNoPad.apply(x, size, stride, dilation, dim, pad)
