import os 
import argparse
import torch

from read_data import DataLoader
from utils import plot_config
from select_gpu import select_gpu
from base_model import BaseModel

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

parser = argparse.ArgumentParser(description="Parser for AutoSF (evaluation)")
parser.add_argument('--task_dir', type=str, default='KG_Data/FB15K237', help='the directory to dataset')
parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
parser.add_argument('--lamb', type=float, default=0.4, help='set weight decay value')
parser.add_argument('--decay_rate', type=float, default=1.0, help='set weight decay value')
parser.add_argument('--n_dim', type=int, default=256, help='set embedding dimension')
parser.add_argument('--thres', type=float, default=0.0, help='threshold for early stopping')
parser.add_argument('--lr', type=float, default=0.7, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=300, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=4096, help='number of training batches')
parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--mode', type=str, default='evaluate', help='which mode this code is running for')
parser.add_argument('--out_file_info', type=str, default='_tune', help='extra string for the output file name')

args = parser.parse_args()

dataset = args.task_dir.split('/')[-1]

directory = 'results'
if not os.path.exists(directory):
    os.makedirs(directory)

torch.cuda.set_device(select_gpu())

args.out_dir = directory
args.perf_file = os.path.join(directory, dataset + '_evaluate.txt')
print('output file name:', args.perf_file)


   
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "4"   
    os.environ["MKL_NUM_THREADS"] = "4"   
    loader = DataLoader(args.task_dir)
    n_ent, n_rel = loader.graph_size()

    train_data = loader.load_data('train')
    valid_data = loader.load_data('valid')
    test_data  = loader.load_data('test')
    n_train = len(train_data[0])
    valid_head_filter, valid_tail_filter, test_head_filter, test_tail_filter = loader.get_filter()

    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data  = [torch.LongTensor(vec) for vec in test_data]

    if dataset == 'WN18':
        args.lr = 0.109
        args.lamb = 0.0003245
        args.n_dim = 1024
        args.decay_rate = 0.991
        args.n_batch = 256
        args.n_epoch = 400
        args.epoch_per_test = 20
        struct = [3,2,1,0,3,3,2,-1,0,2,3,-1,1,0,1,1,2,1,0,1,2,2,0,-1,1,0,2,-1]
    elif dataset == 'FB15K':
        args.lr = 0.704
        args.lamb = 3.49e-5
        args.n_dim = 2048
        args.decay_rate = 0.991
        args.n_batch = 256
        args.n_epoch = 700
        args.epoch_per_test = 50
        struct = [3,2,1,0,1,0,1,1,2,1,0,1]
    elif dataset == 'WN18RR':
        args.lr = 0.471
        args.lamb = 5.92e-05
        args.n_dim = 512
        args.n_batch = 512
        args.decay_rate = 0.99
        args.n_epoch = 300
        args.epoch_per_test = 20
        struct = [1,0,2,3,3,1,1,1,3,0,0,1,0,3,2,-1,1,2,3,-1,1,3,1,-1,0,1,3,-1]
    elif dataset == 'FB15K237':
        args.lr = 0.178
        args.lamb = 0.00252
        args.decay_rate = 0.991
        args.n_batch = 512
        args.n_dim = 2048
        args.n_epoch = 500
        args.epoch_per_test = 25
        struct = [2,1,3,0]
    elif dataset == 'YAGO':
        args.lr = 0.9514
        args.lamb = 0.0002178
        args.n_dim = 1024
        args.decay_rate = 0.991
        args.n_batch = 2048
        args.n_epoch = 400
        args.epoch_per_test = 20
        struct = [0,1,2,3,1,3,1,-1,2,1,3,-1]

    if args.mode == 'tune':
        def run_kge(params):
            args.lr = params['lr']
            args.lamb = 10**params['lamb']
            args.decay_rate = params['decay_rate']
            args.n_batch = params['n_batch']
            args.n_dim = params['n_dim']
            plot_config(args)

            model = BaseModel(n_ent, n_rel, args, struct)
            tester_val = lambda: model.test_link(valid_data, valid_head_filter, valid_tail_filter)
            tester_tst = lambda: model.test_link(test_data,  test_head_filter,  test_tail_filter)
            best_mrr, best_str = model.train(train_data, tester_val, tester_tst)
            with open(args.perf_file, 'a') as f:
                print('structure:', struct, best_str)
                for s in struct:
                    f.write(str(s)+' ')
                f.write(best_str + '\n')
            return {'loss': -best_mrr, 'status': STATUS_OK}

        space4kge = {
            "lr": hp.uniform("lr", 0, 1),
            "lamb": hp.uniform("lamb", -5, 0),
            "decay_rate": hp.uniform("decay_rate", 0.99, 1.0),
            "n_batch": hp.choice("n_batch", [128, 256, 512, 1024]),
            "n_dim": hp.choice("n_dim", [64]),
        }


        trials = Trials()
        best = fmin(run_kge, space4kge, algo=partial(tpe.suggest, n_startup_jobs=30), max_evals=200, trials=trials)

    else:
        plot_config(args)
        model = BaseModel(n_ent, n_rel, args, struct)
        tester_val = lambda: model.test_link(valid_data, valid_head_filter, valid_tail_filter)
        tester_tst = lambda: model.test_link(test_data,  test_head_filter,  test_tail_filter)
        best_mrr, best_str = model.train(train_data, tester_val, tester_tst)

        with open(args.perf_file, 'a') as f:
            print('structure:', struct, best_str)
            for s in struct:
                f.write(str(s)+' ')
            f.write('\t\tbest_performance: ' + best_str + '\n')



