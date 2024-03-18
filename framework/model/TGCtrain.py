import math

import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import TGCDataSet
from model.evaluation import eva
from torch.nn import Linear
import torch.nn.functional as F

FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class TGC:
    def __init__(self, args):
        self.args = args
        self.the_data = args.dataset
        self.file_path = '../data/%s/%s.txt' % (self.the_data, self.the_data)
        self.emb_path = '../emb/%s/%s_TGC_%d.emb'
        self.feature_path = './pretrain/%s_feature.emb' % self.the_data
        self.label_path = '../data/%s/node2label.txt' % self.the_data
        self.labels = self.read_label()
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch
        self.best_acc = 0
        self.best_nmi = 0
        self.best_ari = 0
        self.best_f1 = 0

        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
        self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)
        self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

        self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        kmeans = KMeans(n_clusters=self.clusters, n_init=20)
        _ = kmeans.fit_predict(self.feature)
        self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

        self.v = 1.0
        self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer])
        self.loss = torch.FloatTensor()

    def read_label(self):
        n2l = dict()
        labels = []
        with open(self.label_path, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                n2l[n_id] = l_id
        reader.close()
        for i in range(len(n2l)):
            labels.append(int(n2l[i]))
        return labels

    def kl_loss(self, z, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        the_kl_loss = F.kl_div((q.log()), p, reduction='batchmean')  # l_clu
        return the_kl_loss

    def target_dis(self, emb):
        q = 1.0 / (1.0 + torch.sum(torch.pow(emb.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / tmp_q.sum(0)
        p = (weight.t() / weight.sum(1)).t()

        return p

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)
        s_pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        s_p = self.target_dis(s_pre_emb)
        s_kl_loss = self.kl_loss(s_node_emb, s_p)
        l_node = s_kl_loss

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)  # [b]
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)  # [b,h]
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)  # [b,n]
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss

        l_framework = l_node + l_batch

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(
            dim=1)  # [b]

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)

        total_loss = loss.sum() + l_framework

        return total_loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=1)

            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                self.update(sample_batched['source_node'].type(LType).cuda(),
                            sample_batched['target_node'].type(LType).cuda(),
                            sample_batched['target_time'].type(FType).cuda(),
                            sample_batched['neg_nodes'].type(LType).cuda(),
                            sample_batched['history_nodes'].type(LType).cuda(),
                            sample_batched['history_times'].type(FType).cuda(),
                            sample_batched['history_masks'].type(FType).cuda())

            if self.the_data == 'arxivLarge' or self.the_data == 'arxivPhy' or self.the_data == 'arxivMath':
                acc, nmi, ari, f1 = 0, 0, 0, 0
            else:
                acc, nmi, ari, f1 = eva(self.clusters, self.labels, self.node_emb)

            if nmi > self.best_nmi and epoch > 10:
                self.best_acc = acc
                self.best_nmi = nmi
                self.best_ari = ari
                self.best_f1 = f1

            sys.stdout.write('\repoch %d: loss=%.4f  ' % (epoch, (self.loss.cpu().numpy() / len(self.data))))
            sys.stdout.write('ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)\n' % (acc, nmi, ari, f1))
            sys.stdout.flush()

        print('Best performance: ACC(%.4f) NMI(%.4f) ARI(%.4f) F1(%.4f)' %
              (self.best_acc, self.best_nmi, self.best_ari, self.best_f1))
        self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(str(n_idx) + ' ' + ' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
