import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTMCell

class RNNClassifier(nn.Module):
    """Some Information about RNNClassifier"""
    def __init__(self, vocab_size, embed_size=100, num_layers=2, hidden_size=256, num_classes=2):
        super(RNNClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=256, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = self.embed(x)
        # print(x.shape)
        x = self.lstm(x)
        # print(x.shape)
        # print(x[-1, :, :].shape)
        x = self.fc(x) 
        # print(x.shape)
        return x