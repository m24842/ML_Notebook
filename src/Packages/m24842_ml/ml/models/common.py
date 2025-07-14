import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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

class Butterfly(nn.Module):
    def __init__(self, in_size, out_size, bias=True, complex=False,
                 increasing_stride=True, init='randn', n_blocks=1,
                 device="cpu"):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        self.device = device
        assert n_blocks >= 1
        self.n_blocks = n_blocks
        dtype = torch.get_default_dtype() if not self.complex else {torch.float32: torch.complex64, torch.float64: torch.complex128}[torch.get_default_dtype()]
        twiddle_shape = (self.nstacks, n_blocks, log_n, n // 2, 2, 2)
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['empty', 'randn', 'ortho', 'identity', 'fft', 'ifft']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self._reset_parameters()
        self.to(device)

    def _reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)
        twiddle = self.twiddle
        if self.init is None or self.init == 'empty':
            return
        elif self.init == 'randn':
            # complex randn already has the correct scaling of stddev=1.0
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            if not self.complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                               torch.stack((s, c), dim=-1)), dim=-2))
            else:
                # Sampling from the Haar measure on U(2) is a bit subtle.
                # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.rand((3, ) + twiddle_core_shape) * math.pi * 2
                A = torch.exp(1j * (alpha + psi)) * c
                B = torch.exp(1j * (alpha + chi)) * s
                C = -torch.exp(1j * (alpha - chi)) * s
                D = torch.exp(1j * (alpha - psi)) * c
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((A, B), dim=-1),
                                               torch.stack((C, D), dim=-1)), dim=-2))
        elif self.init == 'identity':
            twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_eye = twiddle_eye.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_eye)
        elif self.init in ['fft', 'ifft']:
            assert self.complex, 'fft/ifft init requires Butterfly to be complex'
            special_fn = (self._fft if self.init == 'fft' else self._ifft)
            b_fft = special_fn(self.n, normalized=True, br_first=self.increasing_stride)
            with torch.no_grad():
                twiddle[:, 0] = b_fft.twiddle
            if self.n_blocks > 1:
                twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
                twiddle_eye = twiddle_eye.expand(*twiddle[:, 1:].shape).contiguous()
                with torch.no_grad():
                    twiddle[:, 1:] = twiddle_eye

    def _complex_reshape(self, x, *shape):
        if not x.is_complex():
            return x.reshape(*shape)
        else:
            return torch.view_as_complex(torch.view_as_real(x).reshape(*shape, 2))
        
    def _fft(self, n, normalized=False, br_first=True):
        """ Construct an nn.Module based on Butterfly that exactly performs the FFT.
        Parameters:
            n: size of the FFT. Must be a power of 2.
            normalized: if True, corresponds to the unitary FFT (i.e. multiplied by 1/sqrt(n))
            br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                    br_first=False corresponds to decimation-in-frequency.
        """
        log_n = int(math.ceil(math.log2(n)))
        assert n == 1 << log_n, 'n must be a power of 2'
        factors = []
        for log_size in range(1, log_n + 1):
            size = 1 << log_size
            exp = torch.exp(-2j * math.pi * torch.arange(0.0, size // 2) / size)
            o = torch.ones_like(exp)
            twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                        torch.stack((o, -exp), dim=-1)), dim=-2)
            factors.append(twiddle_factor.repeat(n // size, 1, 1))
        twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
        if not br_first:  # Take conjugate transpose of the BP decomposition of ifft
            twiddle = twiddle.transpose(-1, -2).flip([2])
        # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
        if normalized:
            twiddle /= math.sqrt(2)
        return Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)

    def _ifft(self, n, normalized=False, br_first=True):
        """ Construct an nn.Module based on Butterfly that exactly performs the inverse FFT.
        Parameters:
            n: size of the iFFT. Must be a power of 2.
            normalized: if True, corresponds to unitary iFFT (i.e. multiplied by 1/sqrt(n), not 1/n)
            br_first: which decomposition of iFFT. True corresponds to decimation-in-frequency.
                    False corresponds to decimation-in-time.
        """
        log_n = int(math.ceil(math.log2(n)))
        assert n == 1 << log_n, 'n must be a power of 2'
        factors = []
        for log_size in range(1, log_n + 1):
            size = 1 << log_size
            exp = torch.exp(2j * math.pi * torch.arange(0.0, size // 2) / size)
            o = torch.ones_like(exp)
            twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                        torch.stack((o, -exp), dim=-1)), dim=-2)
            factors.append(twiddle_factor.repeat(n // size, 1, 1))
        twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
        if not br_first:  # Take conjugate transpose of the BP decomposition of fft
            twiddle = twiddle.transpose(-1, -2).flip([2])
        # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
        if normalized:
            twiddle /= math.sqrt(2)
        else:
            twiddle /= 2
        return Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)
    
    def _butterfly_multiply(self, twiddle, input, increasing_stride=True, output_size=None):
        batch_size, nstacks, input_size = input.shape
        n_blocks = twiddle.shape[1]
        log_n = twiddle.shape[2]
        n = 1 << log_n
        assert twiddle.shape == (nstacks, n_blocks, log_n, n // 2, 2, 2)
        # Pad or trim input to size n
        input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
        output_size = n if output_size is None else output_size
        assert output_size <= n
        output = input.contiguous()
        cur_increasing_stride = increasing_stride
        for block in range(n_blocks):
            for idx in range(log_n):
                log_stride = idx if cur_increasing_stride else log_n - 1 - idx
                stride = 1 << log_stride
                # shape (nstacks, n // (2 * stride), 2, 2, stride)
                t = twiddle[:, block, idx].view(
                    nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
                output_reshape = output.view(
                    batch_size, nstacks, n // (2 * stride), 1, 2, stride)
                output = (t * output_reshape).sum(dim=4)
            cur_increasing_stride = not cur_increasing_stride
        return output.view(batch_size, nstacks, n)[:, :, :output_size]
    
    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = (twiddle[:, :, :log_n, :n // 2] if self.increasing_stride
                       else twiddle[:, :, -log_n:, :n // 2])
            output_size = None
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = self._butterfly_multiply(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.n_blocks - 1) % 2 == 1)
            output = self._butterfly_multiply(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))
    
    def pre_process(self, input):
        # Reshape to (N, in_size)
        input_size = input.size(-1)
        output = self._complex_reshape(input, -1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:  # Take top rows
            output = output[:, :out_size]
        if self.bias is not None:
            output = output + self.bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

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

class DilatedSlidingWindow(autograd.Function):
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
    @autograd.function.once_differentiable
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
