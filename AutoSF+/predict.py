import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD

class Regressor(nn.Module):
    def __init__(self,K):
        super(Regressor, self).__init__()
        #length = 118
        length = K*(K+1)
        hid = 2
        self.layer1 = nn.Linear(length, hid)
        self.layer2 = nn.Linear(hid, 1)
        self.act = nn.ReLU()

    def forward(self, x, dropout=0):
        drop = nn.Dropout(dropout)
        x = drop(x)
        outs = self.layer1(x)
        outs = self.act(outs)
        outs = self.layer2(outs).squeeze()
        return outs

class Predictor(object):
    def __init__(self, topk=5, K=4):
        self.model = Regressor(K).cuda()
        self.loss = nn.L1Loss()
        self.batch_size = 12
        self.topk = topk
        self.optim = SGD(self.model.parameters(), lr=0.01, weight_decay=0.1)

    def evaluate(self, candidates, X, Y, dropout=0.0):
        self.model.train()
        n = len(Y)
        train_x = torch.FloatTensor(np.array(X)).cuda()
        train_y = torch.FloatTensor(np.array(Y)).cuda()
        n_iter = 200
        for i in range(n_iter):
            self.model.zero_grad()
            idx = np.random.choice(n, self.batch_size)
            x_batch = train_x[idx]
            y_batch = train_y[idx]
            y_p = self.model(x_batch, dropout)
            loss = self.loss(y_p, y_batch)
            loss.backward()
            self.optim.step()

        test_X = torch.FloatTensor(np.array(candidates)).cuda()
        self.model.eval()
        scores = self.model(test_X, 0).cpu().data.numpy()
        top_index = scores.argsort()[-self.topk:][::-1]
        #print(scores)

        top_y = scores[top_index]
        #print(top_index, top_y)
        print(top_y)
        return top_index


    def get_scores(self, vectors):
        scores = self.model(vectors)
        return scores.data.numpy()

    def train(self, x, y, batch_size=128, dropout=0.0, n_iter=300):
        n = y.size(0)
        batch_size = max(n // 8, 1)
        for i in range(n_iter):
            self.model.zero_grad()
            idx = np.random.choice(n, batch_size)
            x_batch = x[idx]
            y_batch = y[idx]
            y_p = self.model(x_batch, dropout)
            loss = self.loss(y_p, y_batch)
            loss.backward()
            self.optim.step()
