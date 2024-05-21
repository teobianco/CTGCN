# DynGEM and CTGCN
This repository includes the source code relative to baseline methods DynGEM, CTGCN-C and CTGCN-S. It is based on the source code of paper: [K-Core based Temporal Graph Convolutional Network for Dynamic Graphs](https://ieeexplore.ieee.org/document/9240056).

Basing on the basic methods described in the original source code from repository https://github.com/jhljx/CTGCN, we modified methods DynGEM, CTGCN-CC and CTGCN-S in order to make them semi-spuervised.

# Packages Requirements
- [Python](https://www.python.org/downloads/) >= 3.6
- [Numpy](https://github.com/numpy/numpy) >= 1.18.1
- [Pandas](https://github.com/pandas-dev/pandas) >= 1.0.5
- [Scipy](https://github.com/scipy/scipy) >= 1.5.1
- [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) >= 0.23.1
- [Networkx](https://github.com/networkx/networkx) >= 2.4
- [Pytorch](https://github.com/pytorch/pytorch) == 1.5.1
- torch-scatter == 2.0.5
- torch-sparse == 0.6.6
- torch-spline-conv == 1.2.0
- torch-cluster == 1.5.6
- [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) == 1.6.0

Some binaries of pytorch-geometric related libraries can be found in 
https://pytorch-geometric.com/whl/.

# Directory
    
    CTGCN/    
        baseline/                    (implementation of DynGEM method)  
        config/                      (configuration files and configuration tutorial)
        data/                        (data sets)  
            Amazon_setting1/
                1. format/                 (formatted dynamic graph data)  
                2. embedding/              (embedding results)  
                3. label/                  (community label of nodes)
                CTGCN/                     (intermediate data, i.e. k-core data, random walk data)
                community_detection_score/ (scores in each timestep of community detection task using F1 score, Jaccard score and ONMI as metrics)
                nodes_set/                 (node list file)    
                pred_community/            (communities predicted in each timestep by different methods)
                test_set/
                    train_ratio_0.5/           (test sets for each timestep, using a train ratio = 0.5)
                train_set/
                    train_ratio_0.5/           (training sets for each timestep, using a train ratio = 0.5)
                characteristics.txt        (txt files containing main dataset characteristics)
            facebook/
            ......
        evaluation/                  (evaluation task: community detection and its metrics)  
        preprocessing/               (preprocessing tasks, i.e. k-core decomposition, random walk)
        data_characteristics.py      (compute main characteristics of datasets)
        embedding.py                 (data loader and different kinds of embedding)  
        graph.py                     (dynamic graph generation and scalability data generation)  
        k_means_clustering.py        (perform K-means clustering with embeddings already calculated. You may need this file if you performed community detection with DBSCAN)
        layers.py                    (All layers used in CTGCN)  
        main.py                      (Main file of this project)
        metrics.py                   (Loss function)  
        models.py                    (All models of CTGCN)  
        train.py                     (main file used to train different embedding methods)  
        utils.py                     (utility functions)          

# Functions

We modified the original source code. There are two important functions: **preprocessing** and **community detection**. Thus, the corresponding Python commands are:

1. **Preprocessing**: generate k-core subgraphs and perform random walk.

       python3 main.py --config=config/uci.json --task=preprocessing --method=CTGCN-C

2. **Community Detection**: perform graph embedding and cluster them using K-means in order to perform community detectiona on several dynamic graph data sets.

       python3 main.py --config=config/uci.json --task=embedding --method=CTGCN-C

# Parameter Configurations

All other configuration parameters are saved in configuration files. For more detailed configuration information. We provide detailed parameter configuration tutorials, please refer to [config/README.md](https://github.com/jhljx/CTGCN/tree/master/config). 

# Supported Semi-Supervised Dynamic Graph Embedding Methods

We modified original source code from https://github.com/jhljx/CTGCN to provide Semi-Supervised and community detection oriented version of three Dynamic Graph Embedding methods (the link of each method refers to its original Unsupervised version):

- Deep Embedding Method for Dynamic Graphs (DynGEM)　[\[paper\]](https://arxiv.org/abs/1805.11273)　[\[code\]](https://github.com/jhljx/CTGCN/blob/master/baseline/dynGEM.py)      
- Connective Proximity Preserving Core-based Temporal Graph Convolutional Network (CTGCN-C)   [\[paper\]](https://arxiv.org/abs/2003.09902)   [\[code\]](https://github.com/jhljx/CTGCN)
- Structural Similarity Preserving Core-based Temporal Graph Convolutional Network (CTGCN-S)   [\[paper\]](https://arxiv.org/abs/2003.09902)   [\[code\]](https://github.com/jhljx/CTGCN)

# Supported Data Sets

This project use several data sets in link prediction, node classification and graph centrality prediction tasks. The supported data sets are shown as follows:

| **Data Set** | **Nodes** | **Edges** | **Max Degree** | **Max Core** | **Snapshots** |
|:----:|:----:| :----: | :----: |:----: |:----: |
| UCI | 1899 | 59835 | 198 | 16 | 7 |
| AS  | 6828 | 1947704 | 1458 | 13 | 100 |
| Math | 24740 | 323357 | 231 | 15 | 77 |
| Facebook | 60730 | 607487 | 203 | 9 | 27 |
| Enron | 87036 | 530284 | 1150 | 22 | 38 |
| America-Air | 1190 | 13599 | 238 | 64 | 10 |
| Europe-Air | 399 | 5995 | 202 | 33 | 10 | 

In above data sets, America-Air and Europe-Air are synthetic dynamic graphs, while others are real-world dynamic graphs. Most of the aforementioned graph embedding methods can be trained on an 8G GPU when using UCI, AS, America-Air or Europe-Air data sets. For large-scale graphs such as Facebook and Enron, we recommend you to run those methods on GPU with larger memory or directly train those methods on CPU.

# Notes
1. Origin graph file names must be timestamp format or integer number format, otherwise when training dynamic embedding, sorted(f_list) may return a wrong order of files.
2. Weighted random walk are set as default in the `get_walk_info` function of 'preprocessing/walk_generation.py' file.
3. To use new data is only important to have a `1.format` and `3.label` folders inside their folder. All other folders are generated by the algorithm either during preprocessing (`CTGCN` folder) ore during community detection task
4. Apart from dataset Amazon_setting1, all other datasets contain just `1.format` and `3.label` folders for memory reason
5. To generate the `nodes_set` folder and the file inside, inside each datasets folder, you have to run DynGEM method or instead to add this feature into CTGCN methods
6. The original graph edge data doesn't need to have a reverse edge for each edge, because the graph read functions (`get_sp_adj_mat` and `get_nx_graph` functions in 'utils.py') will add reverse edges automatically. All graph data sets are read by `get_sp_adj_mat` and `get_nx_graph` functions.
7. The original graph file header must be 'from_id, to_id, weight', or you will modify the 'get_nx_graph' function of 'utils.py' file. `get_sp_adj_mat` don't care the concrete header name, as long as the first 2 columns are node indices. If the original graph file has only 2 columns,  `get_sp_adj_mat` function will set edge weights as 1 in the 3rd column. If the original graph file has 3 columns, `get_sp_adj_mat` function will set edge weights as values the 3rd column.

# Reference
- [K-Core based Temporal Graph Convolutional Network for Dynamic Graphs](https://ieeexplore.ieee.org/document/9240056)
