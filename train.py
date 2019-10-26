import os
import argparse
import torch
import time
from read_data import DataLoader
from select_gpu import select_gpu
from base_model import BaseModel

from state import StateSpace
from predict import Predictor
import numpy as np
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="Parser for AutoSF")
parser.add_argument('--task_dir', type=str, default='../KG_Data/FB15K237', help='the directory to dataset')
parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
parser.add_argument('--lamb', type=float, default=0.2, help='set weight decay value')
parser.add_argument('--decay_rate', type=float, default=1.0, help='set learning rate decay value')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--parrel', type=int, default=1, help='set gpu #')
parser.add_argument('--lr', type=float, default=0.5, help='set learning rate')
parser.add_argument('--thres', type=float, default=0.22, help='threshold for early stopping')
parser.add_argument('--n_epoch', type=int, default=1000, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=2048, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=250, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')


args = parser.parse_args()

dataset = args.task_dir.split('/')[-1]

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


def run_model(i, state):
    print('new:', i, state, len(state))
    args.perf_file = os.path.join(directory, dataset + '_perf.txt')
    torch.cuda.empty_cache()
    # sleep to avoid multiple gpu occupy
    time.sleep(10*(i%args.parrel)+1)
    torch.cuda.set_device(select_gpu())

    model = BaseModel(n_ent, n_rel, args, state)
    tester_val = lambda: model.test_link(valid_data, valid_head_filter, valid_tail_filter)
    tester_tst = lambda: model.test_link(test_data, test_head_filter, test_tail_filter)
    best_mrr, best_str = model.train(train_data, tester_val, tester_tst)
    with open(args.perf_file, 'a') as f:
        print('structure:', i, state, '\twrite best mrr', best_str)
        for s in state:
            f.write(str(s) + ' ')
        f.write('\t\tbest_performance: '+best_str)
    torch.cuda.empty_cache()
    return best_mrr


if __name__=='__main__':
    os.environ["OMP_NUM_THREADS"] = "5"
    os.environ["MKL_NUM_THREADS"] = "5"
    mp.set_start_method('forkserver')

    state_obj = StateSpace()
    T = 32                  # train for 1000 iterations
    N = 8                  # number of states for train
    NUM_STATES = 256        # number of states for predict
    N_PREDS = 5

    # config predictor
    pred_obj = [Predictor() for i in range(N_PREDS)]


    perf_file = os.path.join(directory, dataset + '_perf.txt')

    for B in [4,6,8,10,12,14,16]:
        best_score = 0
        num_train = 0

        time_train = 0
        time_filt = 0
        time_pred = 0
        if B == 4:
            # only five candidates which worth evaluation in f^4
            TT = 1
        else:
            TT = T
        for t in range(TT):
            states_cand = []
            matrix_cand = []
            t_filt = time.time()
            counts = 0
            for i in range(NUM_STATES):
                state, matrix, count = state_obj.gen_new_state(B, matrix_cand)
                if state is not None:
                    states_cand.append(state)
                    matrix_cand.append(tuple(matrix))
                counts += count
            print('B=%d Iter %d\tsampled %d candidate state for evaluate' % (B, t+1, len(states_cand)), counts)
            states_cand = np.array(states_cand)
            matrix_cand = np.array(matrix_cand)
            time_filt = time.time() - t_filt

            t_pred = time.time()
            if len(states_cand) < N:
                states_train = states_cand
                matrix_train = matrix_cand
            else:
                scores = []
                features = []
                for state in states_cand:
                    features.append(state_obj.state2srf(state))
                    #features.append(state_obj.state2onehot(state))
                features = torch.FloatTensor(np.array(features))
                for m in range(N_PREDS):
                    scores.append(pred_obj[m].get_scores(features))
                scores = np.mean(np.array(scores), 0)
                top_k = scores.argsort()[-N:][::-1]
                states_train = np.array(states_cand[top_k])
                matrix_train = np.array(matrix_cand[top_k])
                print('top_k states selected', scores[top_k], time.time() - t_pred)
            time_pred += time.time() - t_pred

            # train the selected N models in parallel
            scores = []
            t_train = time.time()
            pool = mp.Pool(processes=args.parrel)
            for i, state in enumerate(states_train):
                score = pool.apply_async(run_model, (num_train, state,))
                num_train += 1
                scores.append(score)
            pool.close()
            pool.join()
            print('~~~~~~~~~~~~~~ parallelly train B=%d finished~~~~~~~~~~~~~~ '%(B), t)
            time_train += time.time() - t_train

            for state, matrix, score in zip(states_train, matrix_train, scores):
                scor = score.get()
                if scor > best_score:
                    best_score = scor
                state_obj.history_matrix[(B-4)//2].append(tuple(matrix))
                state_obj.state_and_score[(B-4)//2].append((tuple(state), scor))
            state_obj.update_good((B-4)//2)
            print('number of models trained:', num_train, 'best score:', best_score)
   
            t_pred = time.time()
            # train the predictor
            state_obj.update_train(perf_file)
            in_x = torch.from_numpy(np.array(state_obj.pred_x, dtype='float32'))
            in_y = torch.from_numpy(np.array(state_obj.pred_y, dtype='float32'))
            idx = np.random.choice(in_y.size(0), 16)
            batch_size = max(in_y.size(0) // 8, 1)
            print('------------ start training predictor ------------', in_x.size(), in_y.size())
            for m in range(N_PREDS):
                n = in_y.size(0)
                idx = np.random.choice(n, n*4//N_PREDS)
                pred_obj[m].train(in_x[idx], in_y[idx], batch_size, 0.3, n_iter=200 * (m+1))
            print('\t............ train predictor finished ............')
            time_pred += time.time() - t_pred

            print('time used:',time_train, time_filt, time_pred, B, t)
    
