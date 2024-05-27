# coding: utf-8
import os
import sys
import json
import torch
import argparse
import numpy as np
from utils import get_supported_methods, assign_free_gpus
import faulthandler
faulthandler.enable()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"


# Parse parameters from the input
def parse_args(args):
    parser = argparse.ArgumentParser(prog='CTGCN', description='K-core based Temporal Graph Convolutional Network')
    parser.add_argument('--config', nargs=1, type=str, help='configuration file path', required=True)
    parser.add_argument('--task', type=str, default='embedding', help='task name which is needed to run', required=True)
    parser.add_argument('--method', type=str, default=None, help='graph embedding method, only used for embedding task')
    return parser.parse_args(args)


# Parse parameters from the json configuration file
def parse_json_args(file_path):
    config_file = open(file_path)
    json_config = json.load(config_file)
    config_file.close()
    return json_config


# Preprocessing task
# CGCN-S and CTGCN-S don't use negative sampling loss, so preprocessing is unnecessary!
def preprocessing_task(method, args):
    from preprocessing import preprocess
    assert method in ['GCN', 'GCN_TG', 'GAT', 'GAT_TG', 'SAGE', 'SAGE_TG', 'GIN', 'GIN_TG', 'PGNN', 'CGCN-C', 'GCRN', 'EvolveGCN', 'CTGCN-C']
    preprocess(method, args[method])


# Embedding task
def embedding_task(method, args):
    print(args)
    assert method in get_supported_methods()

    from baseline.dynAE import dyngem_embedding
    from train import gnn_embedding
    args['has_cuda'] = True if torch.cuda.is_available() else False

    if not args['has_cuda'] and 'use_cuda' in args and args['use_cuda']:
        args['use_cuda'] = False
    if 'use_cuda' in args:
        args['has_cuda'] &= args['use_cuda']
    if not args['has_cuda']:  # Use CPU
        torch.set_num_threads(args['thread_num'])

    if method in ['DynGEM', 'DynAE', 'DynRNN', 'DynAERNN']:
        dyngem_embedding(method, args)
    else:
        gnn_embedding(method, args)


# The main function of the CTGCN project
def main(argv):
    args = parse_args(argv[1:])
    print('args:', args)
    config_dict = parse_json_args(args.config[0])
    # This function assigns free GPUs to the program
    if config_dict[args.task][args.method]['use_cuda']:
        assign_free_gpus(max_gpus=4)
    # pass configuration parameters used in different tasks
    if args.task == 'preprocessing':
        args_dict = config_dict[args.task]
        if args.method is None:
            raise AttributeError('Embedding method parameter is needed for the preprocessing task!')
        preprocessing_task(args.method, args_dict)
    elif args.task == 'embedding':
        args_dict = config_dict[args.task]
        if args.method is None:
            raise AttributeError('Embedding method parameter is needed for the graph embedding task!')
        param_dict = args_dict[args.method]
        embedding_task(args.method, param_dict)
    else:
        raise AttributeError('Unsupported task!')


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(150799)
    np.random.seed(150799)
    # Execute code
    main(sys.argv)
