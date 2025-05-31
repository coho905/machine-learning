from torch import nn, sigmoid, tanh, Tensor
from math import sqrt
import torch

class LSTMCell(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Creates an RNN layer with an LSTM activation function

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in the rnn cell.

        """
        super(LSTMCell, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create and initialize parameters W, V, b as described in the text.
        # remember that the parameters are instance variables

        # W, the input weights matrix has size (n x (4 * m)) where n is
        # the number of input features and m is the hidden size
        # V, the hidden state weights matrix has size (m, (4 * m))
        # b, the vector of biases has size (4 * m)
        k = sqrt(1.0 / self.hidden_size)
        self.W = nn.Parameter(torch.empty(self.vocab_size, (4 * self.hidden_size))) # input weights matrix
        self.V = nn.Parameter(torch.empty(self.hidden_size, (4 * self.hidden_size))) # hidden state weights matrix
        self.b = nn.Parameter(torch.empty(4 * self.hidden_size)) # bias vector
        nn.init.uniform_(self.W, -k, k) # initialize weights across uniform distribution
        nn.init.uniform_(self.V, -k, k) # initialize weights across uniform distribution
        nn.init.uniform_(self.b, -k, k) # initialize bias across uniform distribution
        

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an LSTM layer

        Arguments
        ---------
        x: (Tensor) of size (B x n) where B is the mini-batch size and n is the number of input-features.
            If the RNN has only one layer at each time step, x is the input data of the current time-step.
            In a multi-layer RNN, x is the previous layer's hidden state (usually after applying a dropout)
        h: (Tensor) of size (B x m) where m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (B x m), the cell state of the previous time step

        Return
        ------
        h_out: (Tensor) of size (B x m), the new hidden
        c_out: (Tensor) of size (B x m), he new cell state

        """
        a_k = torch.matmul(x, self.W) + torch.matmul(h, self.V) + self.b # compute the activation vector
        i_k, f_k, o_k, g_k = torch.chunk(a_k, 4, dim=1) # split the activation vector into input, forget, output, and candidate gates
        i_k = sigmoid(i_k) # sigmoid to get input gate
        f_k = sigmoid(f_k) # sigmoid to get forget gate
        o_k = sigmoid(o_k) # sigmoid to get output gate
        c_candidate = tanh(g_k) # tanh to get candidate gate

        c_out = torch.mul(f_k, c) + torch.mul(i_k, c_candidate) # update the cell state with forget and input gates
        h_out = torch.mul(o_k, tanh(c_out)) # return the hidden state by multiplying the output gate with the cell state

        return h_out, c_out # return hidden and cell states


