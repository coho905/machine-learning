from lstm_cell import LSTMCell
from basic_rnn_cell import BasicRNNCell
from torch import nn, stack


class CustomRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1, rnn_type='basic_rnn'):
        """
        Creates an recurrent neural network of type {basic_rnn, lstm_rnn}

        basic_rnn is an rnn whose layers implement a tanH activation function
        lstm_rnn is ann rnn whose layers implement an LSTM cell

        Arguments
        ---------
        vocab_size: (int), the number of unique characters in the corpus. This is the number of input features
        hidden_size: (int), the number of units in each layer of the RNN.
        num_layers: (int), the number of RNN layers at each time step
        rnn_type: (string), the desired rnn type. rnn_type is a member of {'basic_rnn', 'lstm_rnn'}
        """
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.rnn = nn.ModuleList() 
        for i in range(num_layers):
            if rnn_type == "basic_rnn": 
                self.rnn.append(BasicRNNCell(vocab_size, hidden_size)) # basic RNN cell for each layer
            elif rnn_type == "lstm_rnn":
                self.rnn.append(LSTMCell(vocab_size, hidden_size)) # LSTM cell for each layer
            else:
                raise ValueError(f"Unknown RNN type {rnn_type}")
            vocab_size = hidden_size # update the input size for the next layer

        # create a ModuleList self.rnn to hold the layers of the RNN
        # and append the appropriate RNN layers to it
        

    def forward(self, x, h, c):
        """
        Defines the forward propagation of an RNN for a given sequence

        Arguments
        ----------
        x: (Tensor) of size (B x T x n) where B is the mini-batch size, T is the sequence length and n is the
            number of input features. x the mini-batch of input sequence
        h: (Tensor) of size (l x B x m) where l is the number of layers and m is the hidden size. h is the hidden state of the previous time step
        c: (Tensor) of size (l x B x m). c is the cell state of the previous time step if the rnn is an LSTM RNN

        Return
        ------
        outs: (Tensor) of size (B x T x m), the final hidden state of each time step in order
        h: (Tensor) of size (l x B x m), the hidden state of the last time step
        c: (Tensor) of size (l x B x m), the cell state of the last time step, if the rnn is a basic_rnn, c should be
            the cell state passed in as input.
        """

        # compute the hidden states and cell states (for an lstm_rnn) for each mini-batch in the sequence
        outs = []
        b, t, n = x.shape
        h_current = [h[i] for i in range(self.num_layers)] # initialize hidden states for each layer
        c_current = [c[i] for i in range(self.num_layers)] # initialize cell states for each layer
        for timestep in range(t):
            x_t = x[:, timestep, :] # loop through each time step
            for i, layer in enumerate(self.rnn): # loop through each layer
                if self.rnn_type == "lstm_rnn":
                    c_t = c[i] # initiliaze the cell state for current layer
                    h_t, c_t = layer(x_t, h_current[i], c_current[i]) # update the hidden state and cell state for current layer
                    c_current[i] = c_t # update the cell state for current layer
                    h_current[i] = h_t # update the hidden state for current layer
                elif self.rnn_type == "basic_rnn":
                    h_t = layer(x_t, h_current[i]) # update the hidden state for current layer
                    h_current[i] = h_t # update the hidden state for current layer
                x_t = h_t # update the input for the next layer
            outs.append(h_t.unsqueeze(1)) # append the hidden state for last layer
        outs = stack(outs, dim=1) # stack outputs across time steps
        h_final = stack(h_current, dim=0) # stack hidden states across layers (for final timestep) to give to next time step
        c_final = stack(c_current, dim=0) # stack cell states across layers (for final timestep) to give to next time step
        if self.rnn_type == "basic_rnn":
            c_final = c # make cell state the same as input
        return outs, h_final, c_final
