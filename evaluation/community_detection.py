import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import *
from sklearn.manifold import TSNE
import pandas as pd


def evaluate_community_detection(dict_train_communities, dict_communities, embeddings, active_nodes, path_save_communities, path_save_scores, time_step, emb_dim, data_loader, do_dbscan=True):
    """Evaluate community detection using DBSCAN or KMeans clustering algorithm"""
    # Consider just active nodes
    print('Embeddings shape of active nodes is: ', embeddings.shape)
    # Eliminate from dict_communities nodes that have value -1
    dict_communities = {key: value for key, value in dict_communities.items() if value != -1}
    # Create true_labels as a 1-D array
    true_labels = np.array(list(dict_communities.values()))
    # Create an empty dictionary to store lists of keys
    true_communities = {}
    # Iterate through the dictionary
    for key, value in dict_communities.items():
        # If the value is not already a key in result_dict, create an empty list for it
        if value not in true_communities.keys():
            true_communities[value] = []
        # Append the current key to the list associated with the current value
        true_communities[value].append(key)
    # Convert the result dictionary into a list of lists
    true_communities = [v for v in true_communities.values()]
    # Detect communities with DBSCAN or KMEANS
    if do_dbscan:
        pred_labels = get_clusters(embeddings, emb_dim)
    else:
        pred_labels = get_kmeans_clusters(embeddings, dict_train_communities)
    # print('Predicted labels are: ', pred_labels)
    # Create pred_communities as a list of lists excluding labels with value -1
    pred_communities = []
    for i in range(max(pred_labels) + 1):
        pred_communities.append([active_nodes[j] for j in range(len(pred_labels)) if pred_labels[j] == i])
    # Before evaluating results we erase from pred_communities nodes used in training
    # So first we delete from dict_train_communities the nodes that have value -1
    dict_train_communities = {key: value for key, value in dict_train_communities.items() if value != -1}
    # Then we delete from pred_communities the nodes that are in dict_train_communities
    for key, value in dict_train_communities.items():
        for i in range(len(pred_communities)):
            if key in pred_communities[i]:
                pred_communities[i].remove(key)
    # Check if there are empy lists in pred_communities
    pred_communities = [x for x in pred_communities if x]
    # print('Predicted communities after pruning training nodes are ', pred_communities)
    # Define pred_labels_score after pruning training nodes from pred_labels
    pred_labels_score = np.array(list(dict_communities.values()))
    # Evaluate the results
    avgf1, avgjaccard, nmi = eval_scores(pred_communities, true_communities, pred_labels_score, true_labels, tmp_print=True)
    # Save the results
    idx2node = {value: key for key, value in data_loader.node2idx_dict.items()}
    nodes = [idx2node[i] for i in active_nodes]
    modality = 'w' if time_step == 0 else 'a'
    if do_dbscan:
        score_name = 'score'
        pred_name = 'pred_commun'
    else:
        score_name = 'score_kmeans'
        pred_name = 'pred_commun_kmeans'
    with open(path_save_scores + f'/{score_name}.txt', modality) as f:
        f.write("Time {time}: F1 Score: {avgf1} , Jaccard Score: {avgjaccard} , NMI: {nmi}\n".format(time=time_step, avgf1=avgf1, avgjaccard=avgjaccard, nmi=nmi))
    with open(path_save_communities + f'/{pred_name}_timestep{time_step}.txt', 'w') as f:
        for i in range(len(pred_labels)):
            f.write("{node} {label}\n".format(node=nodes[i], label=pred_labels[i]))


def get_clusters(embeddings, emb_dim):
    """Get clusters from a given set of embeddings"""
    struct_layer = emb_dim
    min_samples = int(struct_layer / 2)  # min_samples is dim/2 of the embeddings
    print('min_samples is: ', min_samples)
    eps = infer_epsilon(embeddings, min_samples)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    return clustering.labels_


def get_kmeans_clusters(embeddings, dic_train_communities):
    """Get clusters from a given set of embeddings using KMeans"""
    num_train_comm = len(set(dic_train_communities.values()))
    print('Number of train communities is: ', num_train_comm)
    clustering = KMeans(n_clusters=num_train_comm).fit(embeddings)
    return clustering.labels_


def infer_epsilon(embeddings, min_samples):
    """Get the best DBSCAN parameter epsilon for a given set of embeddings using heuristic search"""
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    distances = np.sort(distances, axis=0)
    distances = distances[:, [0,1]]
    distances[:, 0] = list(range(distances.shape[0]))
    theta = get_data_radiant(distances)
    elbow = find_elbow(distances, theta)
    # Find point of maximum curvature in the sorted k-dist graph
    print('Eps is ', distances[elbow, 1])
    return distances[elbow, 1]


def get_data_radiant(data):
    return np.arctan2(data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min())


def find_elbow(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))
    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)
    # return index of elbow
    return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]


def load_data(dataset, method, i, train_ratio):
    """Load data from a given dataset and method"""
    train_path = f'./data/{dataset}/train_set/train_ratio_{train_ratio}/train_set_timestep{i}.txt'
    test_path = f'./data/{dataset}/test_set/train_ratio_{train_ratio}/test_set_timestep{i}.txt'
    embeddings_path = f'./data/{dataset}/2.embedding/{method}/graph{i}.csv'
    nodes_path = f'./data/{dataset}/nodes_set/nodes.csv'
    path_label = f'./data/{dataset}/3.label/communities{i}.txt'
    # First we want to create a dict to map nodes into their index
    keys = []
    i = 0
    with open(nodes_path, 'r') as file:
        for line in file:
            keys.append(int(line.strip()))
            i += 1
    node2idx_dict = {keys[i]: i for i in range(len(keys))}
    with open(train_path, 'r') as file:
        train_dict = {}
        for line in file:
            comm = list(map(int, line.split()))
            train_dict[node2idx_dict[comm[0]]] = comm[1]
    with open(test_path, 'r') as file:
        test_dict = {}
        for line in file:
            comm = list(map(int, line.split()))
            test_dict[node2idx_dict[comm[0]]] = comm[1]
    active_nodes = sorted(list(set(train_dict.keys()).union(set(test_dict.keys()))))
    embeddings = pd.read_csv(embeddings_path, index_col=0, sep='\t')
    nodes_set = pd.read_csv(nodes_path, names=['node'])
    active_filter = nodes_set.loc[active_nodes, 'node'].tolist()
    embeddings = embeddings.reindex(index=active_filter)
    emb_dim = embeddings.shape[1]
    # Dataloader is an object of a class that contains the node2idx_dict
    dataloader = Dataloader_kmeans(node2idx_dict)
    return train_dict, test_dict, embeddings, active_nodes, dataloader, emb_dim


class Dataloader_kmeans:
    """Dataloader class for KMeans"""
    def __init__(self, node2idx_dict):
        self.node2idx_dict = node2idx_dict
