# Deep Temporal Graph Clustering

This is a comprehensive repository that brings together our work on deep temporal graph clustering, including a series of related papers, open source datasets, and implementations of the TGC base code.

If you find any problems, feel free to contact us: ```mengliuedu@163.com```.

## Related Papers

#### [1] ICLR 2024: Deep Temporal Graph Clustering [[paper](https://arxiv.org/abs/2305.10738)][[code](https://github.com/MGitHubL/Deep-Temporal-Graph-Clustering)]

Authors: Meng Liu, Yue Liu, Ke Liang, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu.

*Best Paper Award of China Computational Power Conference, 2024.*

*Youth Oustanding Paper Award Shortlist of World Artificial Intelligence Conference, 2025.*

*Excellent Poster Award of World Young Scientist Summit, 2024.*

*Open-Source Star of Openatom Foundation, 2025.*

#### [2] TPAMI 2025: Deep Temporal Graph Clustering: A Comprehensive benchmark and Datasets [[paper](https://ieeexplore.ieee.org/abstract/document/11119339)][[code](https://github.com/MGitHubL/BenchTGC)]

Authors: Meng Liu, Ke Liang, Siwei Wang, Xingchen Hu, Sihang Zhou, Xinwang Liu.

*Excellent Paper Award of Guangdong-HongKong-Macao Conference on Image and Graphics, 2025.*

#### [3] TNNLS 2025: Multiview Temporal Graph Clustering [[paper](https://ieeexplore.ieee.org/document/11079275)][[code](https://github.com/MGitHubL/MVTGC)]

Authors: Meng Liu, Ke Liang, Hao Yu, Lingyuan Meng, Siwei Wang, Sihang Zhou, Xinwang Liu.

## Code of TGC

This is the PyTorch version of TGC. We want to provide you with as much usable code as possible.


### Prepare

To run the code, you need prepare datasets and pretrain embeddings:

#### For Datasets

You can download the datasets from [Data4TGC](https://github.com/MGitHubL/Data4TGC) and create "data" folder in the same directory as the "emb" and "framework" folders.

#### For Pre-Training

In ```./framework/pretrain/```, you need run the ```pretrain.py``` to generate pretrain embeddings.

Note that these embeddings are used for TGC training, while the features in the dataset are used for training by any other method.

That is, the pre-training of node2vec is only part of the TGC.

#### For Training

You need create a folder for each dataset in ```./emb/``` to store generated node embeddings.

For example, after training with `Patent` dataset, the node embeddings will be stored in ```./emb/patent/```


### Requirements

```
torch==1.12.1
gensim==3.8.3
networkx==2.8.4
numpy==1.23.5
scipy==1.10.1
scikit-learn==1.1.1
munkres==1.1.4
pandas==1.5.2
```

### Run

For each dataset, create a folder in ```emb``` folder with its corresponding name to store node embeddings, i.e., for arXivAI dataset, create ```./emb/arXivAI```.

For training, run the ```main.py``` in the ```./framework``` folder, all parameter settings have default values, you can adjust them in ```main.py```.


### Test

For test, you have two ways:

(1) In the training process, we evaluate the clustering performance for each epoch.

(2) You can also run the ```clustering.py``` in the ```./framework/experiments``` folder.

Note that the node embeddings in the ```./emb./patent/patent_TGC_200.emb``` folder are just placeholders, you need to run the main code to generate them.

### Parameters

```
| Parameter | Default | Description |
|---|---:|---|
| `--data` | `patent` | Dataset name. The script reads the temporal edge file from `../../data/{data}/{data}.txt` and converts it to an edgelist file. |
| `--input` | `../../data/{data}/{data}.edgelist` | Input edgelist file used by node2vec. This file is generated automatically from the temporal edge file. |
| `--output` | `./{data}_feature.emb` | Output file for the pre-trained node embeddings. |
| `--dimensions` | `128` | Dimensionality of the pre-trained node embeddings. This value should be consistent with `--emb_size` in the TGC training stage. |
| `--walk-length` | `80` | Length of each random walk sampled by node2vec. |
| `--num-walks` | `10` | Number of random walks sampled from each node. |
| `--window-size` | `10` | Context window size used by the Skip-gram model in Word2Vec. |
| `--iter` | `1` | Number of training epochs for Word2Vec. |
| `--workers` | `8` | Number of parallel CPU workers used by Word2Vec. |
| `--p` | `1.0` | Return parameter in node2vec. A smaller value encourages the walk to return to the previous node. |
| `--q` | `1.0` | In-out parameter in node2vec. A smaller value encourages outward exploration, while a larger value biases the walk toward local neighborhoods. |
| `--weighted` | `False` | Whether to use edge weights when reading the edgelist. By default, the graph is treated as unweighted. |
| `--directed` | `False` | Whether to treat the graph as directed during node2vec pre-training. By default, the graph is converted to undirected. |

```

## Cite us

If you feel our work has been helpful, thank you for the citation.

```
@ARTICLE{BenchTGC_ML_TPAMI,
  author={Liu, Meng and Liang, Ke and Wang, Siwei and Hu, Xingchen and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Deep Temporal Graph Clustering: A Comprehensive benchmark and Datasets}, 
  year={2025}
}

@inproceedings{TGC_ML_ICLR,
  author={Liu, Meng and Liu, Yue and Liang, Ke and Tu, Wenxuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  title={Deep Temporal Graph Clustering},
  booktitle={The 12th International Conference on Learning Representations},
  year={2024}
}

@ARTICLE{MVTGC_ML_TNNLS,
  author={Liu, Meng and Liang, Ke and Yu, Hao and Meng, Lingyuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multiview Temporal Graph Clustering}, 
  year={2025},
  pages={1-14}
}
```
