# coding: utf-8
import numpy as np
import pandas as pd
import os
import gc
import time
import torch
from models import MLPClassifier
from utils import check_and_make_path, get_neg_edge_samples


# The base class of embedding
class BaseEmbedding:
    base_path: str
    origin_base_path: str
    embedding_base_path: str
    model_base_path: str
    file_sep: str
    full_node_list: list
    node_num: int
    timestamp_list: list
    has_cuda: bool
    device: torch.device

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model', file_sep='\t', has_cuda=False):
        # file paths
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.embedding_base_path = os.path.abspath(os.path.join(base_path, embedding_folder))
        self.model_base_path = os.path.abspath(os.path.join(base_path, model_folder))
        self.has_cuda = has_cuda
        self.device = torch.device('cuda: 0') if has_cuda else torch.device('cpu')
        self.model = model
        self.loss = loss

        self.file_sep = file_sep
        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)  # node num
        self.timestamp_list = sorted(os.listdir(self.origin_base_path))

        check_and_make_path(self.embedding_base_path)
        check_and_make_path(self.model_base_path)

    def clear_cache(self):
        if self.has_cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()

    def prepare(self, load_model, model_file, classifier_file=None, lr=1e-3, weight_decay=0.):
        classifier = self.classifier if hasattr(self, 'classifier') else None

        if load_model:
            model_path = os.path.join(self.model_base_path, model_file)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
                self.model.eval()
            if classifier_file and classifier:
                classifier_path = os.path.join(self.model_base_path, classifier_file)
                classifier.load_state_dict(torch.load(classifier_path))
                classifier.eval()

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        if classifier:
            classifier = classifier.to(self.device)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        return self.model, self.loss, optimizer, classifier

    def get_batch_info(self, **kwargs):
        pass

    def get_model_res(self, **kwargs):
        pass

    def save_embedding(self, output_list, start_idx):
        if isinstance(output_list, torch.Tensor) and len(output_list.size()) == 2:  # static embedding
            embedding = output_list
            output_list = [embedding]
        # output_list supports two type: list and torch.Tensor(2d or 3d tensor)
        for i in range(len(output_list)):
            embedding = output_list[i]
            timestamp = self.timestamp_list[start_idx + i].split('.')[0]
            df_export = pd.DataFrame(data=embedding.cpu().detach().numpy(), index=self.full_node_list)
            embedding_path = os.path.join(self.embedding_base_path, timestamp + '.csv')
            df_export.to_csv(embedding_path, sep=self.file_sep, header=True, index=True)


# Supervised embedding class(used for node classification)
class SupervisedEmbedding(BaseEmbedding):
    classifier: MLPClassifier

    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, classifier: MLPClassifier, model_folder='model', has_cuda=False):
        super(SupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)
        self.classifier = classifier

    def get_batch_info(self, learning_type, node_labels, edge_labels, edge_list, batch_size, shuffle, train_ratio, val_ratio, test_ratio):
        # consider node classification data
        if learning_type == 'S-node':
            assert node_labels
            timestamp_num = len(node_labels)
            device = node_labels[0].device
            idx_train, label_train, idx_val, label_val, idx_test, label_test = [], [], [], [], [], []
            for i in range(timestamp_num):
                cur_node_labels = node_labels[i]  # tensor
                assert cur_node_labels.shape[1] == 2
                node_indices = torch.randperm(self.node_num, device=device) if shuffle else torch.arange(self.node_num, device=device)
                train_num = int(np.floor(self.node_num * train_ratio))
                val_num = int(np.floor(self.node_num * val_ratio))
                test_num = int(np.floor(self.node_num * test_ratio))

                train_indices = node_indices[: train_num]
                idx_train.append(train_indices)
                label_train.append(node_labels[train_indices, 1])
                val_indices = node_indices[train_num: train_num + val_num]
                idx_val.append(val_indices)
                label_val.append(node_labels[val_indices, 1])
                test_indices = node_indices[train_num + val_num: train_num + val_num + test_num]
                idx_test.append(test_indices)
                label_test.append(node_labels[test_indices, 1])
            return idx_train, label_train, idx_val, label_val, idx_test, label_test
        # consider edge classification data
        elif learning_type == 'S-edge':
            assert edge_labels
            timestamp_num = len(edge_labels)
            device = edge_labels[0].device
            idx_train, label_train, idx_val, label_val, idx_test, label_test = [], [], [], [], [], []
            for i in range(timestamp_num):
                all_edges = edge_labels[i]
                assert all_edges.shape[1] == 3
                all_edge_num = edge_labels[i].shape[0]
                edge_indices = torch.randperm(all_edge_num, device=device) if shuffle else torch.arange(all_edge_num, device=device)
                train_num = int(np.floor(all_edge_num * train_ratio))
                val_num = int(np.floor(all_edge_num * val_ratio))
                test_num = int(np.floor(all_edge_num * test_ratio))

                train_indices = edge_indices[: train_num]
                idx_train.append(all_edges[train_indices, :2].transpose(0, 1))
                label_train.append(edge_labels[train_indices, 2])
                val_indices = edge_indices[train_num: train_num + val_num]
                idx_val.append(all_edges[val_indices, :2].transpose(0, 1))
                label_val.append(edge_labels[val_indices, 2])
                test_indices = edge_indices[train_num + val_num: train_num + val_num + test_num]
                idx_test.append(all_edges[test_indices, :2].transpose(0, 1))
                label_test.append(edge_labels[test_indices, 2])
            return idx_train, label_train, idx_val, label_val, idx_test, label_test
        # consider link prediction
        else:
            timestamp_num = len(edge_list)
            device = edge_list[0].device
            idx_train, label_train, idx_val, label_val, idx_test, label_test = [], [], [], [], [], []
            for i in range(1, timestamp_num):
                assert edge_list[i].shape[0] == 2
                all_edge_num = edge_list[i].shape[1]
                all_edges = edge_list[i].transpose(0, 1).tolist()
                all_edge_dict = dict(zip(map(lambda x: tuple(x), all_edges), np.ones(all_edge_num).astype(np.int)))
                all_edges = np.array(all_edges)
                np.random.shuffle(all_edges)

                train_num = int(np.floor(all_edge_num * train_ratio))
                val_num = int(np.floor(all_edge_num * val_ratio))
                test_num = int(np.floor(all_edge_num * test_ratio))

                train_pos_edges = all_edges[: train_num]
                train_edges = get_neg_edge_samples(train_pos_edges, train_num, all_edge_dict, self.node_num, add_label=False)
                val_pos_edges = all_edges[train_num: train_num + val_num]
                val_edges = get_neg_edge_samples(val_pos_edges, val_num, all_edge_dict, self.node_num, add_label=False)
                test_pos_edges = all_edges[train_num + val_num: train_num + val_num + test_num]
                test_edges = get_neg_edge_samples(test_pos_edges, test_num, all_edge_dict, self.node_num, add_label=False)
                train_edges = torch.tensor(train_edges, device=device).transpose(0, 1).long()
                train_labels = torch.cat([torch.ones(train_num, device=device), torch.zeros(train_num, device=device)])
                val_edges = torch.tensor(val_edges, device=device).transpose(0, 1).long()
                val_labels = torch.cat([torch.ones(val_num, device=device), torch.zeros(val_num, device=device)])
                test_edges = torch.tensor(test_edges, device=device).transpose(0, 1).long()
                test_labels = torch.cat([torch.ones(test_num, device=device), torch.zeros(test_num, device=device)])

                idx_train.append(train_edges)
                idx_val.append(val_edges)
                idx_test.append(test_edges)
                label_train.append(train_labels)
                label_val.append(val_labels)
                label_test.append(test_labels)
            del edge_list
            return idx_train, label_train, idx_val, label_val, idx_test, label_test

    def get_model_res(self, learning_type, adj_list, x_list, edge_list, node_dist_list, batch_indices, model, classifier, hx=None):
        structure_list = None
        if model.method_name in ['CGCN-S', 'CTGCN-S']:
            embedding_list, structure_list = model(x_list, adj_list)
            embedding_list = embedding_list[1:] if learning_type == 'S-link' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = [cls_list, embedding_list, structure_list]
        elif model.method_name == 'VGRNN':
            embedding_list, _, loss_data_list = model(x_list, edge_list, hx)
            embedding_list = embedding_list[1:] if learning_type == 'S-link' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
            loss_input_list.append(cls_list)
        elif model.method_name == 'PGNN':
            from baseline.pgnn import preselect_anchor
            dist_max_list, dist_argmax_list = preselect_anchor(self.node_num, node_dist_list, self.device)
            embedding_list = model(x_list, dist_max_list, dist_argmax_list)
            embedding_list = embedding_list[1:] if learning_type == 'S-link' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = cls_list
        elif model.method_name in ['GCN', 'GAT', 'GIN', 'SAGE', 'GCRN']:
            embedding_list = model(x_list, edge_list)
            embedding_list = embedding_list[1:] if learning_type == 'S-link' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = cls_list
        else:
            embedding_list = model(x_list, adj_list)
            embedding_list = embedding_list[1:] if learning_type == 'S-link' else embedding_list
            cls_list = classifier(embedding_list, batch_indices)
            loss_input_list = cls_list
        output_list = structure_list if model.method_name in ['CGCN-S', 'CTGCN-S'] else embedding_list
        return loss_input_list, output_list, hx

    # edge_list parameter is only used by VGRNN, node_dist_list parameter is only used by PGNN
    # node_labels parameter is used for node classification, edge_labels parameter is used for edge classification
    def learn_embedding(self, adj_list, x_list, node_labels=None, edge_labels=None, edge_list=None, node_dist_list=None, learning_type='S-node', epoch=50, batch_size=1024, lr=1e-3, start_idx=0, weight_decay=0.,
                        train_ratio=0.5, val_ratio=0.3, test_ratio=0.2, model_file='ctgcn', classifier_file='ctgcn_cls', load_model=False, shuffle=True, export=True):
        assert train_ratio + val_ratio + test_ratio <= 1.0
        # prepare model, loss model, optimizer and classifier model
        model, loss_model, optimizer, classifier = self.prepare(load_model, model_file, classifier_file, lr, weight_decay)
        idx_train, label_train, idx_val, label_val, idx_test, label_test = self.get_batch_info(learning_type, node_labels, edge_labels, edge_list, batch_size, shuffle, train_ratio, val_ratio, test_ratio)
        self.clear_cache()
        # time.sleep(100)
        best_acc, best_hx = 0, None
        print('start training!')
        st = time.time()
        for i in range(epoch):
            hx = None  # used for VGRNN
            t1 = time.time()
            loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_train, model, classifier, hx)
            loss_train, acc_train, auc_train = loss_model(loss_input_list, label_train)
            loss_train.backward()
            optimizer.step()  # update gradient
            model.zero_grad()
            # validation
            if i == 0:
                print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()))
            else:
                loss_input_list, output_list, hx = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_val, model, classifier, hx)
                loss_val, acc_val, auc_val = loss_model(loss_input_list, label_val)
                print('Epoch: ' + str(i + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train.item()), 'auc_train: {:.4f}'.format(auc_train),
                      'loss_val: {:.4f}'.format(loss_val.item()), 'acc_val: {:.4f}'.format(acc_val.item()), 'auc_val: {:.4f}'.format(auc_val),  'cost time: {:.4f}s'.format(time.time() - t1))
                # supervised embedding would always save the model with the best performance
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_hx = hx
                    if model_file:
                        torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
                    if classifier_file:
                        torch.save(classifier.state_dict(), os.path.join(self.model_base_path, classifier_file))
            self.clear_cache()
        print('finish training!')

        # load embedding model and classifier model
        if model_file:
            model.load_state_dict(torch.load(os.path.join(self.model_base_path, model_file)))
            model.eval()
        if classifier_file:
            classifier.load_state_dict(torch.load(os.path.join(self.model_base_path, classifier_file)))
            classifier.eval()

        print('start model evaluation!')
        loss_input_list, output_list, _ = self.get_model_res(learning_type, adj_list, x_list, edge_list, node_dist_list, idx_test, model, classifier, best_hx)
        loss_test, acc_test, auc_test = loss_model(loss_input_list, label_test)
        print('Test set results:', 'loss= {:.4f}'.format(loss_test.item()), 'accuracy= {:.4f}'.format(acc_test.item()), 'auc= {:.4f}'.format(auc_test.item()))
        print('finish evaluation!')
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(output_list, start_idx)
        del adj_list, x_list, output_list, model
        self.clear_cache()
        print('training total time: ', cost_time, ' seconds!')
        return cost_time


# Unsupervised embedding class
class UnsupervisedEmbedding(BaseEmbedding):
    def __init__(self, base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder='model', has_cuda=False):
        super(UnsupervisedEmbedding, self).__init__(base_path, origin_folder, embedding_folder, node_list, model, loss, model_folder=model_folder, has_cuda=has_cuda)

    def get_model_res(self, adj_list, x_list, edge_list, node_dist_list, model, batch_indices, hx):
        structure_list = None

        if model.method_name in ['CGCN-S', 'CTGCN-S']:
            embedding_list, structure_list = model(x_list, adj_list)
            loss_input_list = [embedding_list, structure_list, batch_indices]
        elif model.method_name == 'VGRNN':
            embedding_list, hx, loss_data_list = model(x_list, edge_list, hx)
            loss_input_list = loss_data_list
            loss_input_list.append(adj_list)
        elif model.method_name == 'PGNN':
            from baseline.pgnn import preselect_anchor
            dist_max_list, dist_argmax_list = preselect_anchor(self.node_num, node_dist_list, self.device)
            embedding_list = model(x_list, dist_max_list, dist_argmax_list)
            loss_input_list = [embedding_list, batch_indices]
        elif model.method_name in ['GCN', 'GAT', 'SAGE', 'GIN', 'GCRN']:
            embedding_list = model(x_list, edge_list)
            loss_input_list = [embedding_list, batch_indices]
        else:
            embedding_list = model(x_list, adj_list)
            loss_input_list = [embedding_list, batch_indices]
        output_list = structure_list if model.method_name in ['CGCN-S', 'CTGCN-S'] else embedding_list
        return loss_input_list, output_list, hx

    def get_batch_info(self, batch_size):
        batch_num = self.node_num // batch_size
        if self.node_num % batch_size != 0:
            batch_num += 1
        return batch_num

    # edge_list parameter is only used by VGRNN, node_dist_list parameter is only used by PGNN
    def learn_embedding(self, adj_list, x_list, edge_list=None, node_dist_list=None, epoch=50, batch_size=1024, lr=1e-3, start_idx=0, weight_decay=0., model_file='ctgcn', load_model=False, shuffle=True, export=True):
        print('start learning embedding!')
        model, loss_model, optimizer, _ = self.prepare(load_model, model_file, lr=lr, weight_decay=weight_decay)
        batch_num = self.get_batch_info(batch_size)
        all_nodes = torch.arange(self.node_num, device=self.device)
        output_list = []

        st = time.time()
        print('start training!')
        for i in range(epoch):
            node_indices = all_nodes[torch.randperm(self.node_num)] if shuffle else all_nodes  # Tensor
            hx = None  # used for VGRNN
            for j in range(batch_num):
                batch_indices = node_indices[j * batch_size: min(self.node_num, (j + 1) * batch_size)]
                t1 = time.time()
                loss_input_list, output_list, hx = self.get_model_res(adj_list, x_list, edge_list, node_dist_list, model, batch_indices, hx)
                loss = loss_model(loss_input_list)
                loss.backward()
                # gradient accumulation
                if j == batch_num - 1:
                    optimizer.step()  # update gradient
                    model.zero_grad()
                t2 = time.time()
                self.clear_cache()
                print('epoch', i + 1, ', batch num = ', j + 1, ', loss:', loss.item(), ', cost time: ', t2 - t1, ' seconds!')
        print('end training!')
        en = time.time()
        cost_time = en - st

        if export:
            self.save_embedding(output_list, start_idx)
        # if model_file is None, then the model would not be saved
        if model_file:
            torch.save(model.state_dict(), os.path.join(self.model_base_path, model_file))
        del adj_list, x_list, output_list, model
        self.clear_cache()
        print('learning embedding total time: ', cost_time, ' seconds!')
        return cost_time