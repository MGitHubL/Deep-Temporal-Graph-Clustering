# Deep Temporal Graph Clustering

Authors: Meng Liu, Yue Liu, Ke Liang, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu

Paper: https://arxiv.org/abs/2305.10738 (ICLR 2024)

This is the PyTorch version of TGC. We want to provide you with as much usable code as possible.

If you find any problems, feel free to contact us: ```mengliuedu@163.com```.

## Prepare

To run the code, you need prepare datasets and pretrain embeddings:

#### For Datasets

You can download the datasets from [Data4TGC](https://github.com/MGitHubL/Data4TGC) and create "data" folder in the same directory as the "emb" and "framework" folders.

#### For Pre-Training

In ```./framework/pretrain```, you need run the pretrain.py to generate pretrain embeddings.

Note that these embeddings are used for TGC training, while the features in the dataset are used for training by any other method.

That is, the pre-training of node2vec is only part of the TGC.

## Run

For each dataset, create a folder in ```emb``` folder with its corresponding name to store node embeddings, i.e., for arXivAI dataset, create ```./emb/arXivAI```.

3 For training, run the ```main.py``` in the ```./framework``` folder, all parameter settings have default values, you can adjust them in ```main.py```.

4 For test, Run the ```clustering.py``` in the ```./framework/experiments``` folder.

Note that the node embeddings in the ```./emb./patent/patent_TGC_200.emb``` folder are just placeholders, you need to run the main code to generate them.


## Cite us

If you feel our work has been helpful, thank you for the citation.

```
@inproceedings{TGC_ML_ICLR,
  title={Deep Temporal Graph Clustering},
  author={Liu, Meng and Liu, Yue and Liang, Ke and Tu, Wenxuan and Wang, Siwei and Zhou, Sihang and Liu, Xinwang},
  booktitle={The 12th International Conference on Learning Representations},
  year={2024}
}
```
