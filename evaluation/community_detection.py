import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import *
from sklearn.manifold import TSNE


def evaluate_community_detection(dict_train_communities, dict_communities, embeddings, active_nodes, path_save_communities, path_save_scores, time_step, emb_dim, data_loader):
    """Evaluate community detection using DBSCAN"""
    # Consider just active nodes
    # embeddings = embeddings[active_nodes]
    print('Embeddings shape of active nodes is: ', embeddings.shape)
    # Eliminate from dict_communities nodes that have value -1
    dict_communities = {key: value for key, value in dict_communities.items() if value != -1}
    # Create true_labels as a 1-D array
    true_labels = np.array(list(dict_communities.values()))
    # Create true_communities as a list of lists
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
    # Standardize the data
    # embeddings = StandardScaler().fit_transform(embeddings)
    # Plot embeddings
    # plot_embeddings(embeddings, true_labels)
    # Detect communities with DBSCAN
    pred_labels = get_clusters(embeddings, emb_dim)
    # print('Predicted labels are: ', pred_labels)
    # Create pred_communities as a list of lists excluding labels with value -1
    pred_communities = []
    for i in range(max(pred_labels) + 1):
        pred_communities.append([active_nodes[j] for j in range(len(pred_labels)) if pred_labels[j] == i])
    # Some prints
    # print('True communities are ', true_communities)
    # print('Predicted communities are ', pred_communities)
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
    with open(path_save_scores + '/score.txt', modality) as f:
        f.write("Time {time}: F1 Score: {avgf1} , Jaccard Score: {avgjaccard} , NMI: {nmi}\n".format(time=time_step, avgf1=avgf1, avgjaccard=avgjaccard, nmi=nmi))
    with open(path_save_communities + '/pred_commun_timestep{time}.txt'.format(time=time_step), 'w') as f:
        for i in range(len(pred_labels)):
            f.write("{node} {label}\n".format(node=nodes[i], label=pred_labels[i]))


def get_clusters(embeddings, emb_dim):
    """Get clusters from a given set of embeddings"""
    struct_layer = emb_dim
    min_samples = int(struct_layer / 2)  # min_samples is 2 * dim of the embeddings
    print('min_samples is: ', min_samples)
    eps = infer_epsilon(embeddings, min_samples)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
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
    # distances = distances[:, 1]
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


# def plot_embeddings(embeddings, labels_3):
#     # Perform t-SNE projection
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     Y_3 = tsne.fit_transform(embeddings)
#     # Create a scatter plot and color points by their labels
#     fig, ax = plt.subplots(1, 1, figsize=(16, 6))
#     scatter_3 = ax.scatter(Y_3[:, 0], Y_3[:, 1], c=labels_3, cmap='nipy_spectral')
#     ax.set_title('t-SNE time 3')
#     plt.colorbar(scatter_3, ax=ax, label='Labels')
#
#     plt.show()