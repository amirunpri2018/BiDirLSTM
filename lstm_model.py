import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_lstm_layer=1):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layer = num_lstm_layer 
        self.batch_size = None
        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_lstm_layer, 
                            bias=True, 
                            batch_first=True, 
                            dropout=0, 
                            bidirectional=True) # bidirectional makes output 2xhidden size
        self.hidden2out = nn.Linear(self.hidden_dim, self.embedding_dim)


    def forward(self, sentence, noun_position):
        self.batch_size = sentence.shape[0]
        input = Variable(sentence) # shape (batch_size, timestamp, embedding_dim)
        out, (h_n, c_n) = self.lstm(input) # out_shape (batchsize, timstamp/sentlength, hidden_dim*2), h_n last hidden state of each passing, c_n memory cell 
        out = out[torch.arange(self.batch_size), noun_position] # get the hidden of the noun_position [passleft, passright] concatenated in one flat vector  
        out = torch.mean(out.view(self.batch_size, 2, self.hidden_dim), dim=1) # take the avg of the output from two hidden obtained from two way passing
        out = self.hidden2out(out) # add a linear layer to convert size to output size
        out = F.normalize(out, dim=1) # normalize vector to 0mean and 1std vector
        return out # (batch, predicted_target_embedding)
