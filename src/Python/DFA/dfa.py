import torch
from torch import nn
from torch.nn import functional as F
import math

mps = torch.device('mps')
cpu = torch.device('cpu')

class DFAFullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, global_error_dim, activation='relu', last_layer=False):
        super(DFAFullyConnected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.global_error_dim = global_error_dim
        self.last_layer = last_layer
        
        self.linear_input = None
        self.linear_output = None
        self.activation_derivative_output = None

        self.linear = nn.Linear(input_dim, output_dim, device=mps)
        
        activation = activation.lower()
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.activation_derivative = lambda x: (x > 0).float()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
            self.activation_derivative = lambda x: F.sigmoid(x) * (1 - F.sigmoid(x))
        elif activation == 'tanh':
            self.activation = nn.Tanh()
            self.activation_derivative = lambda x: 1 - F.tanh(x) ** 2
        elif activation == 'silu':
            self.activation = nn.SiLU()
            self.activation_derivative = lambda x: F.silu(x) + F.sigmoid(x) * (1 - F.silu(x))
        elif activation == 'cos':
            self.activation = lambda x: torch.cos(x)
            self.activation_derivative = lambda x: -torch.sin(x)
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
            self.activation_derivative = lambda x: torch.cos(x)
        elif activation == 'none':
            self.activation = None
            self.activation_derivative = lambda x: torch.ones_like(x)
        else:
            raise ValueError("Unsupported activation function")
        
        self.feedback = torch.randn(output_dim, global_error_dim, device=mps) if not last_layer else None
    
    @torch.no_grad
    def forward(self, x, calc_grad=True):
        self.linear_input = x
        self.linear_output = self.linear(x)
        activation_output = self.activation(self.linear_output) if self.activation else self.linear_output
        
        self.activation_derivative_output = self.activation_derivative(self.linear_output) if calc_grad else None
        
        return activation_output

    @torch.no_grad
    def backward(self, global_error):
        if self.last_layer:
            da = global_error * self.activation_derivative_output.T
        else:
            if self.activation_derivative_output is None:
                da = torch.matmul(self.feedback, global_error) * self.activation_derivative(self.linear_output).T
            else:
                da = torch.matmul(self.feedback, global_error) * self.activation_derivative_output.T

        dW = torch.matmul(da, self.linear_input)
        db = torch.sum(da, axis=1)
        
        self.linear.weight.grad = dW
        self.linear.bias.grad = db
        
        return dW, db

class DFALayerNorm(nn.Module):
    def __init__(self, dim, global_error_dim, eps=1e-5):
        super(DFALayerNorm, self).__init__()
        self.dim = dim
        self.global_error_dim = global_error_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(dim, device=mps))
        self.beta = nn.Parameter(torch.zeros(dim, device=mps))
        
        # Feedback matrix for DFA
        self.feedback = torch.randn(dim, global_error_dim, device=cpu)

        # Store intermediate values for manual backward pass
        self.input = None
        self.mean = None
        self.var = None
        self.inv_std = None
        self.norm_output = None

    @torch.no_grad
    def forward(self, x, calc_grad=True):
        self.input = x

        # Compute mean and variance
        self.mean = x.mean(dim=-1, keepdim=True)
        self.var = x.var(dim=-1, unbiased=False, keepdim=True)
        self.inv_std = torch.rsqrt(self.var + self.eps)  # 1 / sqrt(var + eps)
        
        # Normalize
        self.norm_output = ((x - self.mean) * self.inv_std)
        
        # Apply affine transformation
        out = self.gamma * self.norm_output + self.beta
        
        return out

    @torch.no_grad
    def backward(self, global_error):
        # Compute DFA error signal
        da = torch.matmul(self.feedback.to(mps), global_error)

        # Compute gradients for normalization step
        dx_hat = da * self.gamma.unsqueeze(-1)
        dvar = torch.sum(dx_hat.T * (self.input - self.mean) * -0.5 * self.inv_std**3, dim=-1, keepdim=True)
        dmean = torch.sum(dx_hat.T * -self.inv_std, dim=-1, keepdim=True) + dvar * torch.mean(-2.0 * (self.input - self.mean), dim=-1, keepdim=True)

        # Compute input gradient
        dx = dx_hat.T * self.inv_std + dvar * 2.0 * (self.input - self.mean) / self.dim + dmean / self.dim

        # Compute gradients for learnable parameters
        dgamma = torch.sum(da.T * self.norm_output, dim=0)
        dbeta = torch.sum(da.T, dim=0)

        # Store gradients
        self.gamma.grad = dgamma
        self.beta.grad = dbeta

        return dx, dgamma, dbeta
    
class StateSpaceNet(nn.Module):
    def __init__(self, io_dim, state_dim):
        super(StateSpaceNet, self).__init__()
        self.io_dim = io_dim
        self.state_dim = state_dim
        
        self.register_buffer('state', None)
        
        self.A_log_gen = nn.Linear(self.io_dim + self.state_dim, self.state_dim * self.state_dim, bias=False, device=mps)
        self.B_gen = nn.Linear(self.io_dim + self.state_dim, self.io_dim * self.state_dim, bias=False, device=mps)
        self.C_gen = nn.Linear(self.io_dim + self.state_dim, self.state_dim * self.io_dim, bias=False, device=mps)
        self.D_gen = nn.Linear(self.io_dim + self.state_dim, self.io_dim * self.io_dim, bias=False, device=mps)
        
        nn.init.normal_(self.A_log_gen.weight, -1e-9, 1e-9)
        nn.init.normal_(self.B_gen.weight, -1e-9, 1e-9)
        nn.init.normal_(self.C_gen.weight, -1e-9, 1e-9)
        nn.init.normal_(self.D_gen.weight, -1e-9, 1e-9)
        self.norm = nn.RMSNorm(self.io_dim, device=mps)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        if self.state is None:
            self.state = torch.zeros((x.size(0), 1, self.state_dim), device=mps)
        input_state_cat = torch.cat([x, self.state], dim=2)
        
        A = -torch.exp(self.A_log_gen(input_state_cat)).reshape(-1, self.state_dim, self.state_dim)
        B = self.B_gen(input_state_cat).reshape(-1, self.io_dim, self.state_dim)
        C = self.C_gen(input_state_cat).reshape(-1, self.state_dim, self.io_dim)
        D = self.D_gen(input_state_cat).reshape(-1, self.io_dim, self.io_dim)
        
        self.state = torch.matmul(self.state, A) + torch.matmul(x, B)
        output = torch.matmul(self.state, C) + torch.matmul(x, D)
        output = self.norm(output + x)
        
        return output.squeeze(1)
    
    def reset(self):
        self.state = None