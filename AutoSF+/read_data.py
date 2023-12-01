import os
import torch
import numpy as np
from itertools import count
from collections import namedtuple, defaultdict


class DataLoader:
    def __init__(self, task_dir):
        self.inPath = task_dir

        with open(os.path.join(self.inPath, "relation2id.txt")) as f:
            tmp = f.readline()
            self.n_rel = int(tmp.strip())

        with open(os.path.join(self.inPath, "entity2id.txt")) as f:
            tmp = f.readline()
            self.n_ent = int(tmp.strip())

        self.train_head, self.train_tail, self.train_rela = self.read_data("train2id.txt")
        self.valid_head, self.valid_tail, self.valid_rela = self.read_data("valid2id.txt")
        self.test_head,  self.test_tail,  self.test_rela  = self.read_data("test2id.txt")

    def read_data(self, filename):
        allList = []
        head = []
        tail = []
        rela = []
        with open(os.path.join(self.inPath, filename)) as f:
            tmp = f.readline()
            total = int(tmp.strip())
            for i in range(total):
                tmp = f.readline()
                h, t, r = tmp.strip().split()
                h, t, r = int(h), int(t), int(r)
                allList.append((h, t, r))

        allList.sort(key=lambda l:(l[0], l[1], l[2]))

        head.append(allList[0][0])
        tail.append(allList[0][1])
        rela.append(allList[0][2])

        for i in range(1, total):
            if allList[i] != allList[i-1]:
                h, t, r = allList[i]
                head.append(h)
                tail.append(t)
                rela.append(r)
        return head, tail, rela

    def graph_size(self):
        return (self.n_ent, self.n_rel)

    def load_data(self, index):
        if index == 'train':
            return self.train_head, self.train_tail, self.train_rela
        elif index == 'valid':
            return self.valid_head, self.valid_tail, self.valid_rela
        else:
            return self.test_head,  self.test_tail,  self.test_rela

    def get_filter(self,):
        all_heads = self.train_head + self.valid_head + self.test_head
        all_tails = self.train_tail + self.valid_tail + self.test_tail
        all_relas = self.train_rela + self.valid_rela + self.test_rela

        heads = defaultdict(lambda: set())
        tails = defaultdict(lambda: set())
        for h, t, r in zip(all_heads, all_tails, all_relas):
            tails[(h, r)].add(t)
            heads[(t, r)].add(h)

        def get_vector(x, all_x):
            v = np.ones(self.n_ent)
            v[list(all_x)] = -1
            v[x] = 1
            return v

        valid_head_filter = []
        valid_tail_filter = []
        for i in range(len(self.valid_head)):
            h = self.valid_head[i]
            t = self.valid_tail[i]
            r = self.valid_rela[i]
            v_h = get_vector(h, heads[(t,r)])
            v_t = get_vector(t, tails[(h,r)])
            valid_head_filter.append(v_h)
            valid_tail_filter.append(v_t)
        valid_head_filter = np.array(valid_head_filter)
        valid_tail_filter = np.array(valid_tail_filter)

        test_head_filter = []
        test_tail_filter = []
        for i in range(len(self.test_head)):
            h = self.test_head[i]
            t = self.test_tail[i]
            r = self.test_rela[i]
            v_h = get_vector(h, heads[(t,r)])
            v_t = get_vector(t, tails[(h,r)])
            test_head_filter.append(v_h)
            test_tail_filter.append(v_t)
        test_head_filter = np.array(test_head_filter)
        test_tail_filter = np.array(test_tail_filter)
        return valid_head_filter, valid_tail_filter, test_head_filter, test_tail_filter

