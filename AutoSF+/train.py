import os
import argparse
import torch
import time
from read_data import DataLoader
from select_gpu import select_gpu
from predict import Predictor
from base_model import BaseModel

from structure import StructSpace
import numpy as np

parser = argparse.ArgumentParser(description="Parser for AutoSF+")
parser.add_argument('--task_dir', type=str, default='KG_Data/FB15K237', help='the directory to dataset')
parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
parser.add_argument('--lamb', type=float, default=0.2, help='set weight decay value')
parser.add_argument('--decay_rate', type=float, default=1.0, help='set learning rate decay value')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--lr', type=float, default=0.5, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=2048, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')


args = parser.parse_args()

dataset = args.task_dir.split('/')[-1]
args.K = 4
args.n_dim = 64
args.optim = 'adagrad'

# WN18RR

if dataset == 'WN18RR':
    args.lr = 0.211797
    args.lamb = 0.00005
    args.n_batch = 128
    args.test_batch_size = 50
    n_epoch = 50

elif dataset == 'WN18':
    args.lr = 0.294506
    args.lamb = 0.001337
    args.n_batch = 2048
    args.test_batch_size = 50
    n_epoch = 60

elif dataset == 'FB15K237':
    args.lr = 0.18821
    args.lamb = 0.003383
    args.n_batch = 1024
    args.test_batch_size = 50
    n_epoch = 50

elif dataset == 'FB15K':
    args.lr = 0.173476
    args.lamb = 0.000013
    args.n_batch = 1024
    args.test_batch_size = 50
    n_epoch = 100


elif dataset == 'YAGO':
    args.lr = 0.201078
    args.lamb = 0.000364
    args.n_batch = 256
    args.n_dim = 32
    args.test_batch_size = 50
    n_epoch = 50

elif dataset == 'DDB14':
    args.lr = 0.630588
    args.lamb = 0.045218
    args.n_batch = 512
    args.n_dim = 64
    args.test_batch_size = 50
    n_epoch = 60
else:
    print(dataset, 'does not exist')
    exit()



if __name__=='__main__':
    torch.cuda.set_device(select_gpu())
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"

    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    args.out_dir = directory
    loader = DataLoader(args.task_dir)
    n_ent, n_rel = loader.graph_size()

    train_data = loader.load_data('train')
    valid_data = loader.load_data('valid')
    test_data = loader.load_data('test')
    n_train = len(train_data[0])
    valid_head_filter, valid_tail_filter, test_head_filter, test_tail_filter = loader.get_filter()

    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data  = [torch.LongTensor(vec) for vec in test_data]

    struct_obj = StructSpace(args.K)
    MAX_POPU = 5
    N_CAND = 100
    N_EVAL = 10
    topk = 8

    perf_file = os.path.join(directory, dataset + args.out_file_info + '.txt')

    basic_struct = struct_obj.populations
    predictor = Predictor(topk=topk, K=args.K)

    model_perf = []
    model_struc = []

    best_mrr = 0
    print('start training initial populations', len(basic_struct))
    init_time = time.time()


    # initialization
    for struct in basic_struct:
        model = BaseModel(n_ent, n_rel, args)
        tester_val = lambda x: model.test_link(valid_data, valid_head_filter, valid_tail_filter, x)

        early_stop = 0
        for epoch in range(n_epoch):
            model.train(train_data, struct)
        mrr, mr, h_1, h_3, h_10 = tester_val(struct)
        out_str = '%s %d \t %.4f %.1f %.4f %.4f %.4f\n' % (str(struct), time.time() - init_time, mrr, mr, h_1, h_3, h_10)
        with open(perf_file, 'a') as f:
            f.write(out_str)
        if mrr > best_mrr:
            best_mrr = mrr
            print('new best arrived:', struct, '\t', mrr, mr, h_1, h_3, h_10)

        struct_obj.get_pred_data(struct, mrr)

        model_perf.append(mrr)
        model_struc.append(tuple(struct))

    print('Finished evaluating initial popolutions.', MAX_POPU, len(model_perf), '\t')

    n_turn = 0
    n_models = len(model_perf)

    # the terminate condition can be set, e.g. while(n_turn<10),while(model_struct<100)
    while(True):
        n_turn += 1
        # generate new candidates
        struct_cand = struct_obj.gen_new_struct(model_struc, N_CAND)
        print('generate new finished')
        features = []
        for struct in struct_cand:
            features.append(struct_obj.gen_features(struct))
        top_idx = predictor.evaluate(features, struct_obj.pred_x, struct_obj.pred_y)
        struct_eval = np.array(struct_cand)[top_idx]
        print('predictor finished', len(struct_eval))

        # train and evaluate the new candidates
        for e, struct in enumerate(struct_eval):
            new_model = BaseModel(n_ent, n_rel, args)
            for epoch in range(n_epoch):
                new_model.train(train_data, struct)
            
            struct_obj.back_ups = struct_obj.back_ups.union(struct_obj.get_equivalence(struct))
            n_models += 1

            tester_val = lambda x:new_model.test_link(valid_data, valid_head_filter, valid_tail_filter, x)
            mrr, mr, h_1, h_3, h_10 = tester_val(struct)
            struct_obj.get_pred_data(struct, mrr)
            out_str = '%s %d \t %.4f %.1f %.4f %.4f %.4f\n' % (str(list(struct)), time.time() - init_time, mrr, mr, h_1, h_3, h_10)
            with open(perf_file, 'a') as f:
                f.write(out_str)

            if mrr > best_mrr:
                best_mrr = mrr
                print('new best arrived:', struct, '\t', mrr, mr, h_1, h_3, h_10)

            if len(model_perf) < MAX_POPU:
                model_perf.append(mrr)
                model_struc.append(tuple(struct))
            elif mrr > min(model_perf):
                idx = np.argmin(model_perf)
                print('\t\tupdate popu:', model_struc[idx], model_perf[idx], '--->', struct, mrr)
                model_perf[idx] = mrr
                model_struc[idx] = struct
            print(n_models, struct, '\t', mrr, mr, h_1, h_3, h_10, '\t', best_mrr)

