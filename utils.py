import random
import numpy as np

def plot_config(args):
    out_str = "\noptim:{} lr:{} lamb:{}, d:{}\n".format(
            args.optim, args.lr, args.lamb, args.n_dim)
    with open(args.perf_file, 'a') as f:
        f.write(out_str)

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        
def gen_struct(num):
    struct = []
    for i in range(num):
        struct.append(random.randint(0,3))
        struct.append(random.randint(0,3))
        struct.append(2*random.randint(0,1)-1)
    return struct

def cal_ranks(probs, label):
    sorted_idx = np.argsort(probs, axis=1)[:,::-1]
    find_target = sorted_idx == np.expand_dims(label, 1)
    ranks = np.nonzero(find_target)[1] + 1
    return ranks

def cal_performance(ranks, topk=10):
    mrr = (1. / ranks).sum() / len(ranks)
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_k = sum(ranks<=topk) * 1.0 / len(ranks)
    return mrr, m_r, h_k
