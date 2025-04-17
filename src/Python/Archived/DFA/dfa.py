import torch
from torch import nn
from torch.nn import functional as F
import math
from opt_einsum import contract

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
        elif activation == 'step':
            self.activation = lambda x: (x > 0).float()
            self.activation_derivative = lambda x: torch.exp(-torch.abs(x))
        elif activation == 'none':
            self.activation = None
            self.activation_derivative = lambda x: torch.ones_like(x)
        else:
            raise ValueError("Unsupported activation function")
        
        self.feedback = torch.randn(output_dim, global_error_dim, device=mps) / math.sqrt(global_error_dim) if not last_layer else None
    
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
            da = global_error * self.activation_derivative_output
        else:
            if self.activation_derivative_output is None:
                da = contract('oe, be, bo -> bo', self.feedback, global_error, self.activation_derivative(self.linear_output), optimize='auto')
            else:
                da = contract('oe, be, bo -> bo', self.feedback, global_error, self.activation_derivative_output, optimize='auto')

        dW = contract('bo, bi -> oi', da, self.linear_input, optimize='auto')
        db = torch.sum(da, axis=0)
        
        self.linear.weight.grad = dW
        self.linear.bias.grad = db
        
        return dW, db