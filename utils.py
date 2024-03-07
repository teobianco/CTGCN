# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import torch
import subprocess
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Check the existence of directory(file) path, if not, create one
def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)


# Get networkx graph object from file path. If the graph is unweighted, then add the 'weight' attribute
def get_nx_graph(file_path, full_node_list, sep='\t'):
    df = pd.read_csv(file_path, index_col=False, names=['from_id', 'to_id'], header=0, sep='\s+')
    if df.shape[1] == 2:
        df['weight'] = 1.0
    graph = nx.from_pandas_edgelist(df, "from_id", "to_id", edge_attr='weight', create_using=nx.Graph)
    graph.add_nodes_from(full_node_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# Get sparse.lil_matrix type adjacent matrix
# Note that if you want to use this function, please transform the sparse matrix type, i.e. sparse.coo_matrix, sparse.csc_matrix if needed!
def get_sp_adj_mat(file_path, full_node_list, sep='\t'):
    node_num = len(full_node_list)
    node2idx_dict = dict(zip(full_node_list, np.arange(node_num)))
    A = sp.lil_matrix((node_num, node_num))
    with open(file_path, 'r') as fp:
        content_list = fp.readlines()
        # ignore header
        for line in content_list:
            line_list = list(map(int, line.split()))
            col_num = len(line_list)
            assert col_num in [2, 3]
            if col_num == 2:
                from_node, to_node, weight = line_list[0], line_list[1], 1
            else:
                from_node, to_node, weight = line_list[0], line_list[1], float(line_list[2])
            from_id = node2idx_dict[from_node]
            to_id = node2idx_dict[to_node]
            # remove self-loop data
            if from_id == to_id:
                continue
            A[from_id, to_id] = weight
            A[to_id, from_id] = weight
    A = A.tocoo()
    return A


# Generate a row-normalized adjacent matrix from a sparse matrix
# If add_eye=True, then the renormalization trick would be used.
# For the renormalization trick, please refer to the "Semi-supervised Classification with Graph Convolutional Networks" paper,
# The paper can be viewed in https://arxiv.org/abs/1609.02907
def get_normalized_adj(adj, row_norm=False):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    p = -1 if row_norm else -0.5

    def inv(x, p):
        if p >= 0:
            return np.power(x, p)
        if x == 0:
            return x
        if x < 0:
            raise ValueError('invalid value encountered in power, x is negative, p is negative!')
        return np.power(x, p)
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum, p).flatten()
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    if not row_norm:
        adj = adj.dot(r_mat_inv)
    adj = adj.tocoo()
    return adj


# Transform a sparse matrix into a torch.sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Transform a sparse matrix into a tuple
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# Generate negative links(edges) in a graph
def get_neg_edge_samples(pos_edges, edge_num, all_edge_dict, node_num, add_label=True):
    neg_edge_dict = dict()
    neg_edge_list = []
    cnt = 0
    while cnt < edge_num:
        from_id = np.random.choice(node_num)
        to_id = np.random.choice(node_num)
        if from_id == to_id:
            continue
        if (from_id, to_id) in all_edge_dict or (to_id, from_id) in all_edge_dict:
            continue
        if (from_id, to_id) in neg_edge_dict or (to_id, from_id) in neg_edge_dict:
            continue
        if add_label:
            neg_edge_list.append([from_id, to_id, 0])
        else:
            neg_edge_list.append([from_id, to_id])
        cnt += 1
    neg_edges = np.array(neg_edge_list)
    all_edges = np.vstack([pos_edges, neg_edges])
    return all_edges


# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# Get formatted number str, which is used for dynamic graph(or embedding) file name. File order is important!
def get_format_str(cnt):
    max_bit = 0
    while cnt > 0:
        cnt //= 10
        max_bit += 1
    format_str = '{:0>' + str(max_bit) + 'd}'
    return format_str


# Print separate lines
def separate(info='', sep='=', num=8):
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)


def get_static_gnn_methods():
    gnn_list = ['GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN', 'CGCN-C', 'CGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_dynamic_gnn_methods():
    gnn_list = ['GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_core_based_methods():
    gnn_list = ['CGCN-C', 'CGCN-S', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_supported_gnn_methods():
    gnn_list = ['GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN', 'CGCN-C', 'CGCN-S', 'GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(gnn_list, np.ones(len(gnn_list), dtype=np.int)))


def get_supported_methods():
    method_list = ['DynGEM', 'DynAE', 'DynRNN', 'DynAERNN', 'TIMERS', 'GCN', 'TgGCN', 'GAT', 'TgGAT', 'SAGE', 'TgSAGE', 'GIN', 'TgGIN', 'PGNN',
                   'CGCN-C', 'CGCN-S', 'GCRN', 'EvolveGCN', 'VGRNN', 'CTGCN-C', 'CTGCN-S']
    return dict(zip(method_list, np.ones(len(method_list), dtype=np.int)))


def split_label_list(labels_list, active_nodes_list, train_path, test_path, data_loader, args, idx):
    train_list = []
    test_list = []
    if args['already_train_test']:
        print("Load train_list and test_list from file")
        for time in range(len(labels_list)):
            # Load train_list and test_list from file
            num_nodes = len(data_loader.full_node_list)
            train_dict = dict()
            test_dict = dict()
            for i in range(num_nodes):
                train_dict[i] = -1
                test_dict[i] = -1
            with open(train_path + '/train_set_timestep{time}.txt'.format(time=idx+time), 'r') as f:
                for line in f:
                    node, label = map(int, line.split())
                    train_dict[data_loader.node2idx_dict[node]] = label
            with open(test_path + '/test_set_timestep{time}.txt'.format(time=idx+time), 'r') as f:
                for line in f:
                    node, label = map(int, line.split())
                    test_dict[data_loader.node2idx_dict[node]] = label
            train_list.append(train_dict)
            test_list.append(test_dict)
    else:
        train_ratio = args['train_ratio']
        idx2node_dict = {value: key for key, value in data_loader.node2idx_dict.items()}
        for time in range(len(labels_list)):
            label_dict = labels_list[time]
            train_dict = dict()
            test_dict = dict()
            values_set = set(label_dict.values())
            values_set.discard(-1)
            values = list(values_set)
            np.random.shuffle(values)
            train_num = int(len(values) * train_ratio)
            train_values = values[:train_num]
            for node in label_dict:
                if label_dict[node] in train_values:
                    train_dict[node] = label_dict[node]
                    test_dict[node] = -1
                else:
                    train_dict[node] = -1
                    test_dict[node] = label_dict[node]
            train_list.append(train_dict)
            test_list.append(test_dict)
            # Filter active nodes in train_dict and test_dict
            save_train = {key: value for key, value in train_dict.items() if (key in active_nodes_list[time] and value != -1)}
            save_test = {key: value for key, value in test_dict.items() if (key in active_nodes_list[time] and value != -1)}
            # Save save_train and save_test to file
            with open(train_path + '/train_set_timestep{time}.txt'.format(time=idx+time), 'w') as f:
                for key, value in save_train.items():
                    f.write("{key} {value}\n".format(key=idx2node_dict[key], value=value))
            with open(test_path + '/test_set_timestep{time}.txt'.format(time=idx+time), 'w') as f:
                for key, value in save_test.items():
                    f.write("{key} {value}\n".format(key=idx2node_dict[key], value=value))
    return train_list, test_list

def split_train_test_new(self, label_dict, frac_train, active_nodes, data_loader, time, train_path='train_set', test_path='test_set'):
    idx2node_dict = {value: key for key, value in data_loader.node2idx_dict.items()}
    train_dict = dict()
    test_dict = dict()
    values_set = set(label_dict.values())
    values_set.discard(-1)
    values = list(values_set)
    np.random.shuffle(values)
    train_num = int(len(values) * frac_train)
    train_values = values[:train_num]
    for node in label_dict:
        if label_dict[node] in train_values:
            train_dict[node] = label_dict[node]
            test_dict[node] = -1
        else:
            train_dict[node] = -1
            test_dict[node] = label_dict[node]
    # Filter active nodes in train_dict and test_dict
    save_train = {key: value for key, value in train_dict.items() if
                  (key in active_nodes and value != -1)}
    save_test = {key: value for key, value in test_dict.items() if
                 (key in active_nodes and value != -1)}
    # Save save_train and save_test to file
    with open(train_path + '/train_set_timestep{time}.txt'.format(time=time), 'w') as f:
        for key, value in save_train.items():
            f.write("{key} {value}\n".format(key=idx2node_dict[key], value=value))
    with open(test_path + '/test_set_timestep{time}.txt'.format(time=time), 'w') as f:
        for key, value in save_test.items():
            f.write("{key} {value}\n".format(key=idx2node_dict[key], value=value))
    return train_dict, test_dict


def mean_n_std(path):
    score_file_path = path
    with open(score_file_path, 'r') as file:
        f1 = []
        jaccard = []
        nmi = []
        for line in file:
            # read after the second :
            splits = line.split(': ')
            f1.append(float(splits[2].split()[0]))
            jaccard.append(float(splits[3].split()[0]))
            nmi.append(float(splits[4].split()[0]))
    # Calculate mean and std
    print('F1 mean: ', np.mean(f1))
    print('F1 std: ', np.std(f1))
    print('Jaccard mean: ', np.mean(jaccard))
    print('Jaccard std: ', np.std(jaccard))
    print('NMI mean: ', np.mean(nmi))
    print('NMI std: ', np.std(nmi))

def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere

    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_to_use
    print(f"Using GPU(s): {gpus_to_use}")
