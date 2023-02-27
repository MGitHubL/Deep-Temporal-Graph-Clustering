import math

import torch
from torch.autograd import Variable
# torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
import numpy as np
import sys
from model.DataSet import TGCDataSet
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
        self.emb_size = args.emb_size
        self.neg_size = args.neg_size
        self.hist_len = args.hist_len
        self.batch = args.batch_size
        self.clusters = args.clusters
        self.save_step = args.save_step
        self.epochs = args.epoch

        self.data = TGCDataSet(self.file_path, self.neg_size, self.hist_len, self.feature_path, args.directed)
        self.node_dim = self.data.get_node_dim()
        self.edge_num = self.data.get_edge_num()
        self.feature = self.data.get_feature()

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=True)
                self.pre_emb = Variable(torch.from_numpy(self.feature).type(FType).cuda(), requires_grad=False)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

                self.enc_1 = Variable((torch.zeros(self.emb_size, 256) + 1.).type(FType).cuda(), requires_grad=True)
                self.enc_2 = Variable((torch.zeros(256, 512) + 1.).type(FType).cuda(), requires_grad=True)
                self.enc_3 = Variable((torch.zeros(512, 1024) + 1.).type(FType).cuda(), requires_grad=True)
                self.z_layer = Variable((torch.zeros(1024, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)

                self.dec_1 = Variable((torch.zeros(self.emb_size, 1024) + 1.).type(FType).cuda(), requires_grad=True)
                self.dec_2 = Variable((torch.zeros(1024, 512) + 1.).type(FType).cuda(), requires_grad=True)
                self.dec_3 = Variable((torch.zeros(512, 256) + 1.).type(FType).cuda(), requires_grad=True)
                self.x_bar_layer = Variable((torch.zeros(256, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)

                self.cluster_layer = Variable((torch.zeros(self.clusters, self.emb_size) + 1.).type(FType).cuda(), requires_grad=True)
                torch.nn.init.xavier_normal_(self.cluster_layer.data)

                kmeans = KMeans(n_clusters=self.clusters, n_init=20)
                _ = kmeans.fit_predict(self.feature)
                self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

                self.v = 1.0
                self.global_cluster = Variable((torch.zeros(self.clusters) + 1.).type(FType).cuda(), requires_grad=True)
                self.batch_weight = math.ceil(self.batch / self.edge_num)

        self.opt = SGD(lr=args.learning_rate, params=[self.node_emb, self.delta, self.cluster_layer, self.enc_1,
                                                      self.enc_2, self.enc_3, self.z_layer, self.dec_1, self.dec_2,
                                                      self.dec_3, self.x_bar_layer, self.global_cluster])
        self.loss = torch.FloatTensor()

    def AE(self, node_emb):
        enc_h1 = F.relu(torch.matmul(node_emb, self.enc_1))
        enc_h2 = F.relu(torch.matmul(enc_h1, self.enc_2))
        enc_h3 = F.relu(torch.matmul(enc_h2, self.enc_3))
        enc_emb = torch.matmul(enc_h3, self.z_layer)

        dec_h1 = F.relu(torch.matmul(enc_emb, self.dec_1))
        dec_h2 = F.relu(torch.matmul(dec_h1, self.dec_2))
        dec_h3 = F.relu(torch.matmul(dec_h2, self.dec_3))
        dec_emb = torch.matmul(dec_h3, self.x_bar_layer)

        return enc_emb, dec_emb

    def kl_loss(self, z, z1, p):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        the_kl_loss = F.kl_div((q.log() + q1.log()), p, reduction='batchmean')
        return the_kl_loss, q.log()

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
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size,-1)
        pre_emb = self.pre_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)

        enc_emb, dec_emb = self.AE(pre_emb)
        p = self.target_dis(s_node_emb)
        s_kl_loss, q = self.kl_loss(s_node_emb, enc_emb, p)
        l_node = s_kl_loss - torch.log(F.mse_loss(dec_emb, pre_emb).sigmoid())

        new_st_adj = torch.cosine_similarity(s_node_emb, t_node_emb)
        res_st_loss = torch.norm(1 - new_st_adj, p=2, dim=0)
        new_sh_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), h_node_emb, dim=2)
        new_sh_adj = new_sh_adj * h_time_mask
        new_sn_adj = torch.cosine_similarity(s_node_emb.unsqueeze(1), n_node_emb, dim=2)
        res_sh_loss = torch.norm(1 - new_sh_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        res_sn_loss = torch.norm(0 - new_sn_adj, p=2, dim=0).sum(dim=0, keepdims=False)
        l_batch = res_st_loss + res_sh_loss + res_sn_loss

        dis = torch.sum(q, dim=0, keepdims=False)
        dis = dis / dis.sum()
        global_cluster = self.global_cluster / self.global_cluster.sum()
        batch_similar = torch.cosine_similarity(global_cluster, dis, dim=0)
        l_intra_batch1 = torch.norm(1 - batch_similar, p=2)
        l_intra_batch2 = torch.abs(1 - self.global_cluster.sum())
        l_graph = l_intra_batch1 + l_intra_batch2

        l_framework = l_node + l_batch + l_graph

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2).neg(), dim=1)

        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (
            Variable(h_time_mask).unsqueeze(2))).sum(dim=1)

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                loss = -torch.log(p_lambda.sigmoid() + 1e-6) - torch.log(
                    n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
        else:
            loss = -torch.log(torch.sigmoid(p_lambda) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambda)) + 1e-6).sum(dim=1)

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
            loader = DataLoader(self.data, batch_size=self.batch, shuffle=True, num_workers=4)

            if epoch == 10:
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, epoch))
            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings(self.emb_path % (self.the_data, self.the_data, epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

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
