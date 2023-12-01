import numpy as np
import random
import itertools

class StructSpace:
    def __init__(self, K):
        self.pred_x = []
        self.pred_y = []
        self.K = K
        self.noise = 0.02
        self.back_ups = set()
        self.signs = self.get_sign()
        self.populations = self.gen_base_struct()

    def get_sign(self,):
        base = [2**i for i in range(self.K)]
        signs = []
        for i in range(2**self.K):
            sign = [1] * self.K
            for j in range(self.K):
                if i & base[j] > 0:
                    sign[j] = -1
            signs.append(sign) 
        return signs 

    def gen_base_struct(self,):
        # generate structures with K nonzero blocks
        indices = list(range(self.K))
        all_struct = [] 
        for perm in list(itertools.permutations(indices)):
            struct = [0] * self.K**2
            for i in range(self.K):
                struct[i*self.K + perm[i]] = i+1
                if not self.filt(struct, self.back_ups):
                    continue
                all_struct.append(struct)
                self.back_ups = self.back_ups.union(self.get_equivalence(struct))
        return all_struct

    def filt(self, struct, H_set):
        matrix = np.reshape(struct, (self.K, self.K))
        
        # not full rank
        if np.linalg.det(matrix) == 0:
            return False

        rela = np.ones((self.K,))

        # all the K components in r should be covered
        for idx in np.nonzero(struct)[0]:
            r = abs(struct[idx]) - 1 
            rela[r] = 0 
        
        if np.sum(rela) > 0:
            return False

        # equivalence
        if tuple(struct) in H_set:
            return False
        return True

    def get_equivalence(self, struct):
        indices = list(range(self.K))
        np_struct = np.array(struct)
        all_struct = set()
        for perm_rel in list(itertools.permutations(indices)):
            for perm_ent in list(itertools.permutations(indices)):
                for sign in self.signs:
                    new_struct = [0] * (self.K**2)
                    for idx in np.nonzero(np_struct)[0]:
                        h = perm_ent[idx // self.K]
                        t = perm_ent[idx % self.K]
                        r = perm_rel[abs(np_struct[idx]) - 1]
                        s = sign[abs(np_struct[idx]) - 1]
                        new_struct[h*self.K+t] = s * np.sign(np_struct[idx]) * (r+1)
                    all_struct.add(tuple(new_struct))

        return all_struct


    def get_pred_data(self, struct, perf, one_hot=False):
        # SRF

        if one_hot:
            features = [0] * (self.K**2 * (self.K+2))
            for idx in np.nonzero(struct)[0]:
                r = abs(struct[idx])
                s = np.sign(struct[idx])
                features[idx*(self.K+2)+r-1] = 1
                if s>0:
                    features[idx * (self.K+2) + self.K] = 1
                else:
                    features[idx * (self.K+2) + self.K+1] = 1
            self.pred_x.append(features)
            self.pred_y.append(perf + self.noise*(2*np.random.random()-1))
            return
        else:
            K = self.K
            values = [list(range(-K, K+1)) for i in range(K)]
    
            SRF = np.zeros((K*(K+1),))
            for assign in itertools.product(*values):
                a = K-np.count_nonzero(assign)
                if a == K:
                    continue
                b = len(list(set(np.abs(assign)) - {0}))
                k = K*a - ((a-1)*a) // 2 + b - 1
                k0 = (K * (K+1)) // 2

                matrix = np.zeros((K, K))
                for idx in np.nonzero(struct)[0]:
                    h = idx // K
                    t = idx % K
                    r = np.sign(struct[idx]) * assign[abs(struct[idx]) - 1]
                    matrix[h][t] = r
                if np.sum(abs(matrix)) == 0:
                    continue
                if np.sum(abs(matrix - matrix.T)) < 0.0001:
                    SRF[k] = 1
                if np.sum(abs(matrix + matrix.T)) < 0.0001 and np.sum(abs(matrix - np.diag(matrix))) > 0:
                    SRF[k+k0] = 1

            self.pred_x.append(SRF)
            self.pred_y.append(perf + self.noise*(2*np.random.random()-1))
            return

    def gen_features(self, struct, one_hot=False):

        if one_hot:
            features = [0] * (self.K**2 * (self.K+2))
            for idx in np.nonzero(struct)[0]:
                r = abs(struct[idx])
                s = np.sign(struct[idx])
                features[idx*(self.K+2)+r-1] = 1
                if s>0:
                    features[idx * (self.K+2) + self.K] = 1
                else:
                    features[idx * (self.K+2) + self.K+1] = 1
            return features

        K = self.K
        values = [list(range(-K, K+1)) for i in range(K)]
    
        SRF = np.zeros((K*(K+1),))
        for assign in itertools.product(*values):
            a = K - np.count_nonzero(assign)
            b = len(list(set(np.abs(assign)) - {0}))
            if a== K:
                continue
            k = K*a - (a-1)*a // 2 + b - 1
            k0 = K * (K+1) // 2

            matrix = np.zeros((K, K))
            for idx in np.nonzero(struct)[0]:
                h = idx // K
                t = idx % K
                r = np.sign(struct[idx]) * assign[abs(struct[idx]) - 1]
                matrix[h][t] = r
            if np.sum(abs(matrix)) == 0:
                continue
            if np.sum(abs(matrix - matrix.T)) < 0.0001:
                SRF[k] = 1
            if np.sum(abs(matrix + matrix.T)) < 0.0001 and np.sum(abs(matrix - np.diag(matrix))) > 0:
                SRF[k+k0] = 1
        return SRF


    def gen_new_struct(self, parents, N_CAND):
        results = []
        current_set = set()
        failed = 0

        while(len(results)<N_CAND and failed<100):
            # crossover
            if np.random.random() < 0.5: 
                p1, p2 = np.random.choice(len(parents), size=(2,), replace = False)
                P1 = parents[p1]
                P2 = parents[p2]
                new_struct = []
                # Cross over
                for i in range(self.K**2):
                    if np.random.random() < 0.5:
                        new_struct.append(P1[i])
                    else:
                        new_struct.append(P2[i])
                # Mutation
                if np.random.random() < 0.2:
                    new_struct = self.mutate(new_struct)
            # mutation
            else:             
                p = np.random.choice(len(parents))
                struct = parents[p]
                new_struct = self.mutate(struct)

            if self.filt(new_struct, self.back_ups.union(current_set)):
                results.append(new_struct)
                new_equiv = self.get_equivalence(new_struct)
                current_set = current_set.union(new_equiv)
                failed = 0
            else: 
                failed += 1
        return results
            
    def mutate(self, old_struct):
        new_struct = list(old_struct)
        for i in range(len(old_struct)):
            if np.random.random() < 2/(self.K**2):
                new_struct[i] = np.random.choice(2*self.K+1) - self.K
        return new_struct

