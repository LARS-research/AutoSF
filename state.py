import numpy as np
import random
import itertools

class StateSpace:
    def __init__(self):
        self.state_count = 0
        self.good_states = [[] for i in range(7)]           # stores top K states
        self.state_and_score = [[] for i in range(7)]     # stores the visited state and score
        self.history_matrix = [[] for i in range(7)]           # stores the matrix
        self.pred_x = []
        self.pred_y = []

    def gen_new_state(self, B, matrix_cand):
        assert B%2 == 0
        count = 0
        while(True):
            if B==4:
                t1 = random.randint(0, 3)
                t2 = random.randint(0, 3)
                t3 = random.randint(0, 3)
                t4 = random.randint(0, 3)
                state = [t1, t2, t3, t4]
            else:
                candidates = self.good_states[(B-6)//2]
                if len(candidates) == 0 or count > 50:
                    return None, None, count
                prev_state = candidates[np.random.choice(len(candidates))]
                r1 = random.randint(0, 3)
                h1 = random.randint(0, 3)
                t1 = random.randint(0, 3)
                s1 = 2*random.randint(0,1) - 1

                r2 = random.randint(0, 3)
                h2 = random.randint(0, 3)
                t2 = random.randint(0, 3)
                s2 = 2*random.randint(0,1) - 1
                state = list(prev_state) + [r1,h1,t1,s1,r2,h2,t2,s2]
            count += 1
            if count >50:          # no further config in this length
                return None, None, count

            matrix = self.state2mat(state, multi=False)
            if matrix is None or self.check_duplicate(state, B, matrix_cand):
                continue
            return state, matrix, count

    def check_duplicate(self, state, B, matrix_cand):
        matrices = self.state2mat(state, multi=True)        # remove duplicate by augmenting with inviriance
        for matrix in matrices:
            if tuple(matrix) in matrix_cand:
                return True
            if tuple(matrix) in self.history_matrix[(B-4)//2]:
                return True
        return False


    def state2srf(self, state):
        vector=  []
        cases = [
            [1,2,3,4], [1,1,2,3], [1,1,1,2], [1,1,2,2], [1,1,1,1], 
            [0,1,2,3], [0,1,1,2], [0,1,1,1], 
            [0,0,1,1], [0,0,1,2], 
            [0,0,0,1]
        ]
        signs = [[1,1,1,1], [1,1,1,-1], [1,1,-1,1], [1,1,-1,-1], [1,-1,1,1,], [1,-1,1,-1], [1,-1,-1,1], [1,-1,-1,-1],
                 [-1,1,1,1], [-1,1,1,-1], [-1,1,-1,1], [-1,1,-1,-1], [-1,-1,1,1,], [-1,-1,1,-1], [-1,-1,-1,1], [-1,-1,-1,-1]]
        for case in cases:
            feat = [0,0]
            for perm in list(itertools.permutations([0,1,2,3])):
                for s in signs:
                    matrix = np.zeros((4,4))
                    for i in range(4):
                        r = perm[i]
                        h = i
                        t = state[i]
                        matrix[h][t] = s[r] * case[r]
                    for i in range(4, len(state)):
                        if i%4 == 0:
                            r = perm[state[i]]
                        elif i%4 == 1:
                            h = state[i]
                        elif i%4 == 2:
                            t = state[i]
                        elif i%4 == 3:
                            matrix[h][t] = state[i] * s[r] * case[r]
                    if np.sum(abs(matrix)) == 0:
                        break
                    if np.sum(abs(matrix - matrix.T)) < 0.0001:
                        feat[0] = 1
                    if np.sum(abs(matrix + matrix.T)) < 0.0001 and np.sum(abs(matrix - np.diag(matrix))) > 0:
                        feat[1] = 1
            vector += feat
        return vector


    def state2mat(self, state, multi=True):
        length = len(state)
        assert length%4 == 0
        if multi:
            matrices = []
            signs = [[1,1,1,1], [1,1,1,-1], [1,1,-1,1], [1,1,-1,-1], [1,-1,1,1,], [1,-1,1,-1], [1,-1,-1,1], [1,-1,-1,-1],
                    [-1,1,1,1], [-1,1,1,-1], [-1,1,-1,1], [-1,1,-1,-1], [-1,-1,1,1,], [-1,-1,1,-1], [-1,-1,-1,1], [-1,-1,-1,-1]]
            for perm_rel in list(itertools.permutations([0,1,2,3])):
                for perm_ent in list(itertools.permutations([0,1,2,3])):
                    for s in signs:
                        matrix = np.zeros((4,4), dtype='int')
                        for i in range(4):
                            r = perm_rel[i] + 1
                            h = perm_ent[i]
                            t = perm_ent[state[i]]
                            matrix[h][t] = s[r-1] * r
                        for i in range(4, length):
                            if i%4 == 0:
                                r = perm_rel[state[i]] + 1
                            elif i%4 == 1:
                                h = perm_ent[state[i]]
                            elif i%4 == 2:
                                t = perm_ent[state[i]]
                            elif i%4 == 3:
                                matrix[h][t] = s[r-1] * state[i] * r
                        matrices.append(tuple(matrix.reshape(-1)))
            return matrices
        else:
            matrix = np.zeros((4,4), dtype='int')
            for i in range(4):
                matrix[i][state[i]] = i+1
            for i in range(4, length):
                if i%4 == 0:
                    r = state[i] + 1
                elif i%4 == 1:
                    h = state[i]
                elif i%4 == 2:
                    t = state[i]
                elif i%4 == 3:
                    if matrix[h][t] != 0 :
                        return None
                    matrix[h][t] = state[i] * r
            for i in range(4):
                if np.sum(abs(matrix[:,i])) == 0:
                    return None
                if np.sum(abs(matrix[i,:])) == 0:
                    return None

            return tuple(matrix.reshape(-1))

    def state2onehot(self, state, evaluate=True):
        length = len(state)
        assert length%4 == 0
        vector = [0] * (16*6)
        for i in range(4):
            r = i
            h = i
            t = state[i]
            vector[h*24 + t*6 + r] = 1          # a_ij
            vector[h*24 + t*6 + 4+1] = 1        # sign
        for i in range(4, length):
            if i%4 == 0:
                r = state[i]
            elif i%4 == 1:
                h = state[i]
            elif i%4 == 2:
                t = state[i]
            elif i%4 == 3:
                sign = max(state[i], 0)
                vector[h*24 + t*6 + r] = 1
                vector[h*24 + t*6 + 4+sign] = 1
        return vector

    def update_good(self, B_idx):
        topk = 8
        goods = []
        if len(self.state_and_score[B_idx]) <= topk:
            for tup in self.state_and_score[B_idx]:
                goods.append(list(tup[0]))
        else:
            sort_tup = sorted(self.state_and_score[B_idx], key=lambda x:x[1],reverse=True)
            for j in range(topk):
                goods.append(list(sort_tup[j][0]))
        self.good_states[B_idx] = goods
    
    def update_train(self, filename):
        self.pred_x = []
        self.pred_y = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip().split()
                state = []
                flag = True
                for c in l:
                    if c[0] == 'b':
                        flag = False
                        continue
                    if flag:
                        state.append(int(c))
                    else:
                        mrr = float(c)
                        break
                self.pred_x.append(self.state2srf(state))
                #self.pred_x.append(self.state2onehot(state))
                self.pred_y.append(mrr)
            
