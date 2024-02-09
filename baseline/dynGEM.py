# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from baseline.dynAE import MLP, RegularizationLoss

# DynGEM: Deep Embedding Method for Dynamic Graphs. For more information, please refer to https://arxiv.org/abs/1805.11273
# We refer to the DynGEM tensorflow source code https://github.com/palash1992/DynamicGEM, and implement a pytorch version of DynGEM
# Author: jhljx
# Email: jhljx8918@gmail.com


# DynGEM class
class DynGEM(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    method_name: str
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, output_dim, n_units=None, bias=True, **kwargs):
        super(DynGEM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.method_name = 'DynGEM'

        self.encoder = MLP(input_dim, output_dim, n_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred


# Loss used for DynGEM
class DynGEMLoss(nn.Module):
    alpha: float
    beta: float
    regularization: RegularizationLoss

    def __init__(self, alpha, beta, gamma, nu1, nu2):
        super(DynGEMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        assert len(input_list) == 11
        xi_pred, x_i, penalty_i = input_list[0], input_list[1], input_list[2]
        xj_pred, x_j, penalty_j = input_list[3], input_list[4], input_list[5]
        hx_i, hx_j, edge_weight = input_list[6], input_list[7], input_list[8]
        labels = input_list[9]  # Dizionario
        nodes = input_list[10]

        node_num = xj_pred.shape[1]
        xi_loss = torch.mean(torch.sum(torch.square((xi_pred - x_i) * penalty_i[:, 0:node_num]), dim=1) / penalty_i[:, node_num])
        xj_loss = torch.mean(torch.sum(torch.square((xj_pred - x_j) * penalty_j[:, 0:node_num]), dim=1) / penalty_j[:, node_num])
        hx_loss = torch.mean(torch.sum(torch.square(hx_i - hx_j), dim=1) * edge_weight)
        # Part for label loss
        labels = torch.tensor([labels[node_id] for node_id in nodes])
        # labels = torch.tensor(labels.values())
        grouped_hx_i = []
        for label in torch.unique(labels):
            if label != -1:
                mask = labels == label  # Maschera booleana per selezionare i nodi con la label corrente
                grouped_hx_i.append(hx_i[mask])
        grouped_hx_i = torch.cat(grouped_hx_i, dim=0)
        diff = grouped_hx_i.unsqueeze(0) - grouped_hx_i.unsqueeze(1)
        label_loss = torch.sum(diff ** 2) / diff.numel()
        # label_loss = 0
        # # Concat hx_i and hx_j
        # hx = torch.cat((hx_i, hx_j), dim=0)
        # # Merge dictionaries label_i and label_j
        # label = label_i.extend(label_j)  #VERIFICARE SE FUNZIONA PER I DIZIONARI
        # for el in label.values():
        #     if el != -1:
        #         ids = [k for k, v in label.items() if v == el]
        #         for i in range(len(ids)):
        #             for j in range(i + 1, len(ids)):
        #                 label_loss += torch.square(hx[ids[i]] - hx[ids[j]])
        # label_loss = torch.mean(label_loss)
        reconstruction_loss = xi_loss + xj_loss + self.alpha * hx_loss
        label_loss = self.gamma * label_loss
        regularization_loss = self.regularization(model)
        return reconstruction_loss + label_loss + regularization_loss


# Batch generator used for DynGEM
class DynGEMBatchGenerator:
    node_list: list
    node_num: int
    batch_size: int
    beta: float
    shuffle: bool
    has_cuda: bool

    def __init__(self, node_list, batch_size, beta, frac_train, shuffle=True, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.beta = beta
        self.shuffle = shuffle
        self.has_cuda = has_cuda
        self.frac_train = frac_train

    def generate(self, graph: sp.lil_matrix, label_dict):
        if isinstance(graph, list):
            assert len(graph) == 1
            graph = graph[0]
        rows, cols, values = sp.find(graph)
        element_num = rows.shape[0]
        element_indices = np.arange(element_num)
        batch_num = element_num // self.batch_size
        if element_num % self.batch_size != 0:
            batch_num += 1

        if self.shuffle:
            np.random.shuffle(element_indices)
        counter = 0
        while True:
            batch_indices = element_indices[self.batch_size * counter: min(element_num, self.batch_size * (counter + 1))]
            xi_batch = torch.from_numpy(graph[rows[batch_indices], :].toarray()).float()
            xi_batch = xi_batch.cuda() if self.has_cuda else xi_batch
            xj_batch = torch.tensor(graph[cols[batch_indices], :].toarray(), device=xi_batch.device).float()
            yi_batch = torch.ones(xi_batch.shape, device=xi_batch.device)  # penalty tensor for x_i
            yj_batch = torch.ones(xj_batch.shape, device=xi_batch.device)  # penalty tensor for x_j
            value_batch = torch.tensor(values[batch_indices], device=xi_batch.device).unsqueeze(1).float()  # [batch_size * 1]
            label_batch = dict()
            batch_nodes = []
            # print('Len batch indices: ', len(batch_indices))
            # print('Shape rows[batch_indices]: ', rows[batch_indices].shape)
            for i in rows[batch_indices]:
                label_batch[i] = label_dict[0][i]
                batch_nodes.append(i)
            # label_batch = {i: label_dict[i] for i in batch_indices}
            num_el_to_hide = len(label_batch.values()) * (1 - self.frac_train)
            # Randomly change values of num_el_to_hide elements of label batch to -1
            for i in range(int(num_el_to_hide)):
                label_batch[np.random.choice(list(label_batch.keys()))] = -1
            yi_batch[xi_batch != 0] = self.beta
            yj_batch[xj_batch != 0] = self.beta
            xi_degree_batch = torch.sum(xi_batch, dim=1).unsqueeze(1)  # [batch_size * 1]
            xj_degree_batch = torch.sum(xj_batch, dim=1).unsqueeze(1)  # [batch_size * 1]

            yi_batch = torch.cat((yi_batch, xi_degree_batch), dim=1)
            yj_batch = torch.cat((yj_batch, xj_degree_batch), dim=1)
            counter += 1
            yield [xi_batch, xj_batch], [yi_batch, yj_batch, value_batch, label_batch, batch_nodes]

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(element_indices)
                counter = 0


# Batch Predictor used for DynGEM
class DynGEMBatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(self, graph, model, batch_indices, counter, embedding_mat, x_pred):
        if isinstance(graph, list):
            assert len(graph) == 1
            graph = graph[0]
        x_batch = torch.tensor(graph[batch_indices, :].toarray()).float()
        x_batch = x_batch.cuda() if self.has_cuda else x_batch
        hx_batch, x_pred_batch = model(x_batch)
        if counter:
            embedding_mat = torch.cat((embedding_mat, hx_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = hx_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph):
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(self.batch_size * counter, self.batch_size * (counter + 1))
            embedding_mat, x_pred = self.get_predict_res(graph, model, batch_indices, counter, embedding_mat, x_pred)

            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(graph, model, remain_indices, counter, embedding_mat, x_pred)
        return embedding_mat, x_pred
