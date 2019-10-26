import torch
import torch.nn as nn

class KGEModule(nn.Module):
    def __init__(self, n_ent, n_rel, args, struct):
        super(KGEModule, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.args = args
        self.struct = struct
        self.lamb = args.lamb
        self.loss = torch.nn.Softplus().cuda()
        self.ent_embed = nn.Embedding(n_ent, args.n_dim)
        self.rel_embed = nn.Embedding(n_rel, args.n_dim)
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)

    def forward(self, head, tail, rela, dropout=True):
        head = head.view(-1)
        tail = tail.view(-1)
        rela = rela.view(-1)

        head_embed = self.ent_embed(head)
        tail_embed = self.ent_embed(tail)
        rela_embed = self.rel_embed(rela)

        # get f = h' R t

        pos_trip = self.test_trip(head_embed, rela_embed, tail_embed)

        neg_tail = self.test_tail(head_embed, rela_embed)
        neg_head = self.test_head(rela_embed, tail_embed)

        max_t = torch.max(neg_tail, 1, keepdim=True)[0]
        max_h = torch.max(neg_head, 1, keepdim=True)[0]

        # multi-class loss: negative loglikelihood
        loss = - 2*pos_trip + max_t + torch.log(torch.sum(torch.exp(neg_tail - max_t), 1)) +\
               max_h + torch.log(torch.sum(torch.exp(neg_head - max_h), 1))
        self.regul = torch.sum(rela_embed**2)

        return torch.sum(loss)

    def test_trip(self, head, rela, tail):
        vec_hr = self.get_hr(head, rela)
        scores = torch.sum(vec_hr * tail, 1)
        return scores

    def test_tail(self, head, rela):
        vec_hr = self.get_hr(head, rela)
        tail_embed = self.ent_embed.weight
        scores = torch.mm(vec_hr, tail_embed.transpose(1,0))
        return scores

    def test_head(self, rela, tail):
        vec_rt = self.get_rt(rela, tail)
        head_embed = self.ent_embed.weight
        scores = torch.mm(vec_rt, head_embed.transpose(1,0))
        return scores

    def get_hr(self, head, rela):
        idx = tuple(self.struct)
        length = self.args.n_dim // 4
        h1 = head[:, :length]
        r1 = rela[:, :length]

        h2 = head[:, 1*length:2*length]
        r2 = rela[:, 1*length:2*length]

        h3 = head[:, 2*length:3*length]
        r3 = rela[:, 2*length:3*length]

        h4 = head[:, 3*length:4*length]
        r4 = rela[:, 3*length:4*length]

        hs = [h1, h2, h3, h4]
        rs = [r1, r2, r3, r4]

        vs = [0, 0, 0, 0]
        vs[idx[0]] = h1*r1
        vs[idx[1]] = h2*r2
        vs[idx[2]] = h3*r3
        vs[idx[3]] = h4*r4
        
        res_B = (len(idx)-4)//4
        for b_ in range(1, res_B+1):
            base = 4*b_
            vs[idx[base+2]] += rs[idx[base+0]] * hs[idx[base+1]] * int(idx[base+3])
        return torch.cat(vs, 1)

    def get_rt(self, rela, tail):
        idx = tuple(self.struct)
        length = self.args.n_dim // 4
        t1 = tail[:, :length]
        r1 = rela[:, :length]

        t2 = tail[:, 1*length:2*length]
        r2 = rela[:, 1*length:2*length]

        t3 = tail[:, 2*length:3*length]
        r3 = rela[:, 2*length:3*length]

        t4 = tail[:, 3*length:4*length]
        r4 = rela[:, 3*length:4*length]

        ts = [t1, t2, t3, t4]
        rs = [r1, r2, r3, r4]

        vs = [r1*ts[idx[0]], r2*ts[idx[1]], r3*ts[idx[2]], r4*ts[idx[3]]]
        
        res_B = (len(idx)-4) // 4
        for b_ in range(1, res_B+1):
            base = 4*b_
            vs[idx[base+1]] += rs[idx[base+0]] * ts[idx[base+2]] * int(idx[base+3])
        return torch.cat(vs, 1)
