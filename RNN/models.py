import torch.nn as nn
from torch.autograd import Variable

class RNN_lower(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(RNN_lower, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.encoder.weight.data.uniform_(-0.1, 0.1)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        return output, hidden

    def init_hidden(self, bsz): #(h, c)
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


class RNN_upper(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        super(RNN_upper, self).__init__()
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.nlayers = nlayers
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))

class RNN_local(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(RNN_local, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, output):
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))
