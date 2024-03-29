import torch
import numpy as np

from utils import batch_by_size, cal_ranks, cal_performance
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ExponentialLR
from models import KGEModule

class BaseModel(object):
    def __init__(self, n_ent, n_rel, args):
        self.model = KGEModule(n_ent, n_rel, args)
        self.model.cuda()

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.time_tot = 0
        self.args = args

        if args.optim=='adam' or args.optim=='Adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        elif args.optim=='adagrad' or args.optim=='Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)
        self.scheduler = ExponentialLR(self.optimizer, self.args.decay_rate)

    def train(self, train_data, struct):
        head, tail, rela = train_data
        n_train = len(head)
        n_batch = self.args.n_batch

        rand_idx = torch.randperm(n_train)
        head = head[rand_idx].cuda()
        tail = tail[rand_idx].cuda()
        rela = rela[rand_idx].cuda()


        for h, t, r in batch_by_size(n_batch, head, tail, rela, n_sample=n_train):
            self.model.zero_grad()
            loss = self.model.forward(h, t, r, struct)
            loss += self.args.lamb * self.model.regul
            loss.backward()
            self.optimizer.step()
            self.prox_operator()
            
        self.scheduler.step()

    def train_many(self, train_data, all_struct):
        head, tail, rela = train_data
        n_train = len(head)
        n_batch = self.args.n_batch


        rand_idx = torch.randperm(n_train)
        head = head[rand_idx].cuda()
        tail = tail[rand_idx].cuda()
        rela = rela[rand_idx].cuda()


        for h, t, r in batch_by_size(n_batch, head, tail, rela, n_sample=n_train):
            self.model.zero_grad()
            sidx = np.random.choice(len(all_struct))
            struct = all_struct[sidx]
            loss = self.model.forward(h, t, r, struct)
            loss += self.args.lamb * self.model.regul
            loss.backward()
            self.optimizer.step()
            self.prox_operator()
            
        self.scheduler.step()

    def prox_operator(self,):
        for n, p in self.model.named_parameters():
            if 'ent' in n:
                X = p.data.clone()
                Z = torch.norm(X, p=2, dim=1, keepdim=True)
                Z[Z<1] = 1
                X = X/Z
                p.data.copy_(X.view(self.n_ent, -1))

    def test_link(self, test_data, head_filter, tail_filter, struct):
        heads, tails, relas = test_data
        batch_size = self.args.test_batch_size
        num_batch = len(heads) // batch_size + int(len(heads)%batch_size>0)

        head_probs = []
        tail_probs = []
        for i in range(num_batch):
            start = i * batch_size
            end = min( (i+1)*batch_size, len(heads))
            batch_h = heads[start:end].cuda()
            batch_t = tails[start:end].cuda()
            batch_r = relas[start:end].cuda()

            h_embed = self.model.ent_embed(batch_h)
            r_embed = self.model.rel_embed(batch_r)
            t_embed = self.model.ent_embed(batch_t)

            head_scores = torch.sigmoid(self.model.test_head(r_embed, t_embed, struct)).data
            tail_scores = torch.sigmoid(self.model.test_tail(h_embed, r_embed, struct)).data

            head_probs.append(head_scores.data.cpu().numpy())
            tail_probs.append(tail_scores.data.cpu().numpy())

        head_probs = np.concatenate(head_probs) * head_filter
        tail_probs = np.concatenate(tail_probs) * tail_filter
        head_ranks = cal_ranks(head_probs, label=heads.data.numpy())
        tail_ranks = cal_ranks(tail_probs, label=tails.data.numpy())
        h_mrr, h_mr, h_h1, h_h3, h_h10 = cal_performance(head_ranks)
        t_mrr, t_mr, t_h1, t_h3, t_h10 = cal_performance(tail_ranks)
        mrr = (h_mrr + t_mrr) / 2
        mr = (h_mr + t_mr) / 2
        h1 = (h_h1 + t_h1) / 2
        h3 = (h_h3 + t_h3) / 2
        h10 = (h_h10 + t_h10) / 2
        return round(mrr,4), round(mr,1), round(h1,4), round(h3,4), round(h10,4)

