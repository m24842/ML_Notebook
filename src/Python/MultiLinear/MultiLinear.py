import torch
import torch.nn as nn
import opt_einsum as oe
import string

class MultiLinear(nn.Module):
    def __init__(self, in_shape, out_shape, pairings=None, bias=True):
        super(MultiLinear, self).__init__()
        self.original_out_shape = out_shape
        
        self.num_modes = max(len(in_shape), len(out_shape))
        self.in_shape = (1,) * (self.num_modes - len(in_shape)) + in_shape
        self.out_shape = (1,) * (self.num_modes - len(out_shape)) + out_shape
        
        # Transformations for each mode
        if pairings is None:
            pairings = list(range(self.num_modes))
        else:
            if len(pairings) != self.num_modes:
                raise ValueError("Length of pairings must match the number of modes")
            if sorted(pairings) != list(range(self.num_modes)):
                raise ValueError("Pairings must be a permutation of 0 ... num_modes-1")
        self.pairings = pairings
        
        self.transforms = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(self.out_shape[i], self.in_shape[self.pairings[i]]))
            for i in range(self.num_modes)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(*self.original_out_shape))
        else:
            self.bias = None

        # Einsum configuration
        in_subscript = ''.join(string.ascii_lowercase[:self.num_modes])
        out_subscript = ''.join(string.ascii_uppercase[:self.num_modes])
        trans_subscripts = ','.join(f'{out}{inp}' for inp, out in zip(in_subscript, out_subscript))
        self.einsum_str = f'...{in_subscript},{trans_subscripts} -> ...{out_subscript}'
        
        # Default precompiled tensor contraction
        self.compile(batch_size=1)

    def forward(self, x):
        x = x.reshape(-1, *self.in_shape)

        y = self.contract_expr(x, *self.transforms)
        
        if self.num_modes > len(self.original_out_shape):
            batch_dims = y.shape[:-self.num_modes]
            y = y.reshape(*batch_dims, *self.original_out_shape)
        
        if self.bias is not None:
            y = y + self.bias
        
        return y

    def compile(self, batch_size):
        # Precompile the tensor contraction to optimize batch processing
        transform_shapes = tuple(param.shape for param in self.transforms)
        self.contract_expr = oe.contract_expression(
            self.einsum_str,
            *((batch_size,) + self.in_shape,),
            *transform_shapes,
            optimize='optimal',
            memory_limit=None,
        )