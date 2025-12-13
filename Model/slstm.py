"""
Scalar LSTM (sLSTM) implementation
Based on the paper's Equation (14) and xLSTM architecture
"""

import torch
import torch.nn as nn
import math


class sLSTMCell(nn.Module):
    """
    Scalar LSTM cell with exponential gating.
    
    Based on the paper's Equation (14):
    - Uses exponential activation for input and forget gates
    - Implements stabilization through normalization
    """
    
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super(sLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        
        # Forget gate
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)
        self.b_f = nn.Parameter(torch.ones(hidden_size))  # Initialize to 1
        
        # Output gate
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # Cell input (modulated input)
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.uniform_(weight, -std, std)
    
    def forward(self, x_t, h_prev, c_prev, n_prev):
        """
        Forward pass of sLSTM cell.
        
        Based on Equation (14) from the paper:
        i_t = exp(W_xi @ X_t + W_hi @ H_{t-1} + b_i)
        f_t = exp(W_xf @ X_t + W_hf @ H_{t-1} + b_f) [or sigmoid]
        o_t = sigmoid(W_xo @ X_t + W_ho @ H_{t-1} + b_o)
        u_t = tanh(W_xc @ X_t + W_hc @ H_{t-1} + b_c)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ u_t
        N_t = f_t ⊙ n_{t-1} + i_t
        H_t = o_t ⊙ tanh(C_t / N_t)
        
        Args:
            x_t: Input at time t, shape (B, input_size)
            h_prev: Previous hidden state, shape (B, hidden_size)
            c_prev: Previous cell state, shape (B, hidden_size)
            n_prev: Previous normalizer state, shape (B, hidden_size)
            
        Returns:
            h_t: New hidden state
            c_t: New cell state
            n_t: New normalizer state
        """
        # Input gate - Use sigmoid for numerical stability instead of exp
        # Original paper uses exp, but sigmoid is more stable for training
        i_t = torch.sigmoid(self.W_xi(x_t) + self.W_hi(h_prev) + self.b_i)
        
        # Forget gate - Use sigmoid for numerical stability  
        f_t = torch.sigmoid(self.W_xf(x_t) + self.W_hf(h_prev) + self.b_f)
        
        # Output gate (sigmoid activation)
        o_t = torch.sigmoid(self.W_xo(x_t) + self.W_ho(h_prev) + self.b_o)
        
        # Modulated input (tanh activation)
        u_t = torch.tanh(self.W_xc(x_t) + self.W_hc(h_prev) + self.b_c)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * u_t
        
        # Update normalizer state
        n_t = f_t * n_prev + i_t
        
        # Clamp normalizer to prevent it from becoming too large
        n_t = torch.clamp(n_t, min=1e-6, max=1e6)
        
        # Compute hidden state with normalization
        h_t = o_t * torch.tanh(c_t / n_t)  # n_t is already clamped
        
        return h_t, c_t, n_t


class sLSTM(nn.Module):
    """
    Scalar LSTM module that processes sequences.
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate between layers
        """
        super(sLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build LSTM layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(sLSTMCell(layer_input_size, hidden_size))
        
        # Dropout layer
        if num_layers > 1 and dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, x, hidden=None):
        """
        Forward pass through sLSTM.
        
        Args:
            x: Input sequence, shape (B, T, input_size)
            hidden: Initial hidden state (optional)
                    Tuple of (h_0, c_0, n_0) for each layer
                    
        Returns:
            output: Output sequence, shape (B, T, hidden_size)
            hidden: Final hidden state (h_n, c_n, n_n)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)
        
        # Unpack hidden states
        h_states, c_states, n_states = hidden
        
        # Process sequence through layers
        layer_output = x
        new_h_states = []
        new_c_states = []
        new_n_states = []
        
        for layer_idx, layer in enumerate(self.layers):
            h_t = h_states[layer_idx]
            c_t = c_states[layer_idx]
            n_t = n_states[layer_idx]
            
            layer_outputs = []
            
            # Process each time step
            for t in range(seq_len):
                h_t, c_t, n_t = layer(layer_output[:, t, :], h_t, c_t, n_t)
                layer_outputs.append(h_t.unsqueeze(1))
            
            # Stack outputs
            layer_output = torch.cat(layer_outputs, dim=1)  # (B, T, hidden_size)
            
            # Apply dropout between layers
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            # Store final states
            new_h_states.append(h_t)
            new_c_states.append(c_t)
            new_n_states.append(n_t)
        
        # Stack hidden states
        new_h_states = torch.stack(new_h_states, dim=0)  # (num_layers, B, hidden_size)
        new_c_states = torch.stack(new_c_states, dim=0)
        new_n_states = torch.stack(new_n_states, dim=0)
        
        return layer_output, (new_h_states, new_c_states, new_n_states)
    
    def _init_hidden(self, batch_size, device):
        """Initialize hidden states."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        n_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0, n_0)

