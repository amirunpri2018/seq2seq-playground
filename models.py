import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=self.dropout, bidirectional=False)

    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        outputs, hidden = self.gru(input_seqs, hidden)
        return outputs, hidden


class Attn(nn.Module):
    """The Attention Module."""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, 1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        # Calculate energy for each encoder output
        attn_energies = torch.t(self.score(
            hidden.expand_as(encoder_outputs), encoder_outputs))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = torch.sum(hidden * encoder_output, 2)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.sum(hidden * energy, 2)
            return energy

        elif self.method == 'concat':
            # TODO: This part seems to have some bugs
            energy = self.attn(torch.cat((hidden, encoder_output), 2))
            energy = torch.sum(self.v.expand_as(hidden) * energy, 2)
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        batch_size = input_seq.size(0)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(input_seq, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.relu(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
