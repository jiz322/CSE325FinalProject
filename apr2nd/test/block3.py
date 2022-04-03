from block2 import *
#-_-_-__-____---

from torch import embedding, nn
# There is really nothing to be stored in this object.
# -- But wait, how about self.rnn and self.fc?
# -- NB: NOW, I assume that the nn keep weights from the inherentance,
# -- And these functions as LSTM and FC will use these weight correctly
class LSTMScoreAssigner(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        """
        :param input_dim: size of the vocabulary (number of unique tokens)
        :param output_dim: number of unique POS tags 
        :param emb_dim: embedding dimensionality of each token
        :param hid_dim: number of hidden neurons of a hidden state/cell
        :param n_layers: number of RNN layers (2 for faster training)
        :param dropout: dropout rate between 0 and 1at the embedding layer and rnn
        :param bidirectional: 1 if use bidirectional and 0 if don't
        """
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        # before output, there is a dropout (except the last layer)
        
        # -- Comment: this is a part of the analysis (last part of this hw)
        # -- It feels like I have no control on this bidirectional since it is part of nn library.
        if bidirectional == 0:
            self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, dropout=dropout)
            self.fc = nn.Linear(hid_dim, output_dim)
            self.num_directions = 1
        elif bidirectional == 1:
            self.rnn = nn.LSTM(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, dropout=dropout, bidirectional=True)
            self.fc = nn.Linear(hid_dim * 2, output_dim)
            self.num_directions = 2

        self.dropout = nn.Dropout(dropout)

    # -- COMMENT
    # -- The src means sourse, which is a 2d array batch_size by sentence_len, it is a big 2d tensor
    # -- NBBBBB: How to turn the POSTaggedDataset into a big 2d tensor see test_p2 line 4-7
    def forward(self, src):
        """

        :param src: a [batch_size, sentence_len] array.
                     Each row is a sequence of word indices and each column represents a position in the sequence.
        :return: the predicted logits at each position. 
        """
        # -- src : a list of sentence tensors
        # -- logit : a tensor having length of self.output_dim
        
        
        ### Your codes go here (20 points) ###

        # Step 1: turn token indices into dense vector,
        # so that embedded is of shape (batch_size, sentence_len, emb_dim)
        src = self.embedding(src)
        # Step 2: rnn maps the tensor (batch_size, sentence_len, emb_dim) to
        # outputs = a tensor (batch_size, sentence_len, hid_dim)
        # hidden = a tensor (batch_size, sentence_len, hid_dim)
        # cell = a tensor (batch_size, sentence_len, hid_dim)
        
        # See library LSTM to continuw
        # Maybe find examples of LSTM
        
        # # Construct a h_0 and c_0
        # h_0 = 0
        # c_0 = 0
        
        # outputs, (hidden, cell) = self.rnn(self.embedding(src, (h_0, c_0)))
        # --- Come back from office hour
        output = self.rnn(src)[0]
        # The  self.rnn(src)[1] is a tuple of (h_n, c_n)
        # -- Think: is c_0 and h_0 necessary? I guess no for now.

        # Step 3: map the output tensor to a logit tensor of shape (batch_size, sentence_len, number_of_POS_tags)
        logit = self.fc(output)
        return torch.sum(logit, dim=1)/logit.shape[1]

