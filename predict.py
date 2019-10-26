import torch.nn as nn
import numpy as np
from torch.optim import SGD

class Regressor(nn.Module):
    def __init__(self,):
        super(Regressor, self).__init__()
        length = 22
        self.layer1 = nn.Linear(length, 2)
        self.layer2 = nn.Linear(2, 1)
        self.act = nn.ReLU()
        #self.act = nn.Tanh()

    def forward(self, x, dropout=0):
        drop = nn.Dropout(dropout)
        x = drop(x)
        outs = self.layer1(x)
        outs = self.act(outs)
        outs = self.layer2(outs).squeeze()
        return outs

class Predictor(object):
    def __init__(self, n_inp=16*6):
        self.n_inp = n_inp
        self.model = Regressor()
        self.loss = nn.L1Loss()
        #self.batch_size = 128
        self.optim = SGD(self.model.parameters(), lr=0.01, weight_decay=0.1)

    def get_scores(self, vectors):
        scores = self.model(vectors)
        return scores.data.numpy()

    def train(self, x, y, batch_size=128, dropout=0.0, n_iter=100):
        n = y.size(0)
        for i in range(n_iter):
            self.model.zero_grad()
            idx = np.random.choice(n, batch_size)
            x_batch = x[idx]
            y_batch = y[idx]
            y_p = self.model(x_batch, dropout)
            loss = self.loss(y_p, y_batch)
            loss.backward()
            self.optim.step()
