import torch
import torch.nn as nn
import numpy as np

class KGEModule(nn.Module):
    def __init__(self, n_ent, n_rel, args):
        super(KGEModule, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.args = args
        self.lamb = args.lamb
        self.loss = torch.nn.Softplus().cuda()
        self.ent_embed = nn.Embedding(n_ent, args.n_dim)
        self.rel_embed = nn.Embedding(n_rel, args.n_dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)

    def forward(self, head, tail, rela, struct, dropout=True):
        head = head.view(-1)
        tail = tail.view(-1)
        rela = rela.view(-1)

        head_embed = self.ent_embed(head)
        tail_embed = self.ent_embed(tail)
        rela_embed = self.rel_embed(rela)

        # get f = h' R t

        pos_trip = self.test_trip(head_embed, rela_embed, tail_embed, struct)

        neg_tail = self.test_tail(head_embed, rela_embed, struct)
        neg_head = self.test_head(rela_embed, tail_embed, struct)

        max_t = torch.max(neg_tail, 1, keepdim=True)[0]
        max_h = torch.max(neg_head, 1, keepdim=True)[0]

        # multi-class loss: negative loglikelihood
        loss = - 2*pos_trip + max_t + torch.log(torch.sum(torch.exp(neg_tail - max_t), 1)) +\
               max_h + torch.log(torch.sum(torch.exp(neg_head - max_h), 1))
        self.regul = torch.sum(rela_embed**2)

        return torch.sum(loss)
    
    def test_trip(self, head, rela, tail, struct):
        vec_hr = self.get_hr(head, rela, struct)
        scores = torch.sum(vec_hr * tail, 1)
        return scores

    def test_tail(self, head, rela, struct):
        vec_hr = self.get_hr(head, rela, struct)
        tail_embed = self.ent_embed.weight
        scores = torch.mm(vec_hr, tail_embed.transpose(1,0))
        return scores

    def test_head(self, rela, tail, struct):
        vec_rt = self.get_rt(rela, tail, struct)
        head_embed = self.ent_embed.weight
        scores = torch.mm(vec_rt, head_embed.transpose(1,0))
        return scores

    def get_hr(self, head, rela, struct):
        # combination of h_embed and r_embed
        nnz = np.nonzero(struct)[0]

        hs = torch.chunk(head, 4, dim=-1)
        rs = torch.chunk(rela, 4, dim=-1)

        vs = [
                torch.zeros(hs[0].size()).cuda(),
                torch.zeros(hs[1].size()).cuda(),
                torch.zeros(hs[2].size()).cuda(),
                torch.zeros(hs[3].size()).cuda()
             ]
        for n in nnz:
            vs[n%4] += np.sign(float(struct[n])) * hs[n//4] * rs[abs(struct[n]) - 1]
        return torch.cat(vs, 1)

    def get_rt(self, rela, tail, struct):
        # combination of r_embed and t_embed
        nnz = np.nonzero(struct)[0]

        ts = torch.chunk(tail, 4, dim=-1)
        rs = torch.chunk(rela, 4, dim=-1)

        vs = [
                torch.zeros(ts[0].size()).cuda(),
                torch.zeros(ts[1].size()).cuda(),
                torch.zeros(ts[2].size()).cuda(),
                torch.zeros(ts[3].size()).cuda()
             ]
        for n in nnz:
            vs[n//4] += np.sign(float(struct[n])) * ts[n%4] * rs[abs(struct[n]) - 1]
        return torch.cat(vs, 1)


