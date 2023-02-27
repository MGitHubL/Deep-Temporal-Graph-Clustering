import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from evaluation import evaluation


def test_external_score(emb_path, label_path, k):
    n2l = dict()
    with open(label_path, 'r') as reader:
        for line in reader:
            parts = line.strip().split()
            n_id, l_id = int(parts[0]), int(parts[1])
            n2l[n_id] = l_id

    node_emb = dict()
    with open(emb_path, 'r') as reader:
        reader.readline()
        for line in reader:
            embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
            node_id = embeds[0]
            if node_id in n2l:
                node_emb[node_id] = embeds[1:]

    Y = []
    X = []
    n2l_list = sorted(n2l.items(), key=lambda x: x[0])
    for (the_id, label) in n2l_list:
        Y.append(label)
        X.append(node_emb[the_id])

    model = KMeans(n_clusters=k, n_init=20)
    cluster_id = model.fit_predict(X)
    center = model.cluster_centers_
    acc, nmi, ari, f1 = evaluation(Y, cluster_id)
    print('ACC: %f, NMI: %f, ARI: %f, F1: %f' % (acc, nmi, ari, f1))


if __name__ == '__main__':
    data = 'arxivAI'
    model = 'TGC_200'

    emb_path = '../../emb/%s/%s_%s.emb' % (data, data, model)
    label_path = '../../data/%s/node2label.txt' % (data)
    ex_dict = {'arxivAI': 5, 'dblp': 10}

    k = ex_dict[data]
    test_external_score(emb_path, label_path, k)
