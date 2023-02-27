from torch.utils.data import Dataset
import numpy as np
import sys
import random


class TGCDataSet(Dataset):
    def __init__(self, file_path, neg_size, hist_len, feature_path, directed, transform=None):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform
        self.feature_path = feature_path

        self.max_d_time = -sys.maxsize

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()
        self.edge_num = 0

        with open(file_path, 'r') as infile:
            for line in infile:
                self.edge_num += 1
                parts = line.split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                d_time = float(parts[2])
                self.node_set.update([s_node,t_node])

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time))

                if d_time > self.max_d_time:
                    self.max_d_time = d_time

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0
                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

        self.node_dim = len(self.node_set)

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,),dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,),dtype=np.int32)
        idx = 0

        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_node_dim(self):
        return self.node_dim

    def get_edge_num(self):
        return self.edge_num

    def get_feature(self):
        node_emb = dict()
        with open(self.feature_path, 'r') as reader:
            reader.readline()
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                node_id = embeds[0]
                node_emb[node_id] = embeds[1:]
            reader.close()
        feature = []
        for i in range(len(node_emb)):
            feature.append(node_emb[i])

        return np.array(feature)

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.

        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id],self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling()

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'neg_nodes': neg_nodes,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
