# Configuration

Here we provide program configurations in this project. This project has two functions: **preprocessing** and **community detection**

Then configurations for each function will be introduced. We also give the detailed parameter usage guide.

## Preprocessing
The parameters used in the **preprocessing** task is shown as follows. Note that If the graph embedding model doesn't use the unsupervised **negative sampling** loss (as for DynGEM and CTGCN-S) then the preprocessing process is **unnecessary**!

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| core_folder | str |  k-core subgraphs folder path relative to `base_path` <br> (**optional**, value: can be null) |
| node_file | str | node file path relative to `base_path` |
| walk_pair_folder | str | random walk node co-occurring pairs file folder path relative to `base_path` |
| node_freq_folder | str | node frequency file folder path relative to `base_path` |
| file_sep | str |  file separator for all files (i.e. '\t') |
| generate_core | bool | whether to generate k-core subgraph files or not <br> (**optional**, value: true or false) |
| run_walk | bool | whether to run random walk or not <br> (**optional**, value: true or false) |
| weighted | bool | weighted graph or not, then this will determine weighted random walk or not |
| walk_time | int | random walk times for each node |
| walk_length | int | random walk length for each node |
| worker | int | CPU multiprocessing thread number <br> (default: -1, means don't use multiprocessing) |

Note that **optional** parameters can be omitted in the configuration file if this optional parameter is not needed. 

We also provide a bool parameter `generate_core` to control the generation of k-core subgraphs. K-core subgraphs will be generated only and if only `core_file` is not null and `generate_core` is true. If you only want to execute random walk in the preprocessing task, you can set `generate_core` as false for CTGCN-C (because its `core_file` is not null), while you can set `core_file` as null for other methods as they don't need k-core subgraphs.

## Community Detection

The community detection task contains the execution of several graph embedding methods. The parameters of all supported graph embedding methods are shown below. 

For a DynGEM graph embedding method, its parameter set is the union of **Common parameters** and **DynamicGEM parameters**

For a GNN based graph embedding method (CTGCN-C and CTGCN-S), its parameter set is the union of **Common parameters**, **CTGCN-C and CTGCN-S parameters** and **Learning parameters**(optional)

### Common parameters
Common parameters are input parameters used by all supported graph embedding methods.

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| base_path | str | base path of a data set |
| origin_folder | str | graph folder path relative to `base_path` |
| core_folder | str | k-core subgraph folder path relative to `base_pah` <br> (**optional**, value: can be null) |
| embed_folder | str | embedding folder path relative to `base_path` |
| model_folder | str | graph embedding model folder path relative to `base_path` | 
| model_file | str | model file name <br> (value: **can be null**, then model won't be saved!) |
| node_file | str | node file path relative to `base_path` |
| file_sep | str | file separator for all files (i.e. '\t') |
| start_idx | int | start timestamp index of dynamic graphs  |
| end_idx | int | end timestamp index of dynamic graphs <br> (default: -1, means the last timestamp, it supports negative values) |
| duration | int | The timestamp length of the input |
| embed_dim | int | node embedding dimension |
| use_cuda | bool | whether or not to use GPU for calculation |
| thread_num | int | the thread number for training if CPU is used |
| epoch | int | training epoch number of a graph embedding model |
| lr | float | learning rate of the optimizer |
| batch_size | int | batch size of a data batch <br> (value: > 0, no other constraints) |
| load_model | bool | whether or not to load graph embedding model for training |
| shuffle | bool | whether or not to shuffle the data for training |
| export | bool | whether or not to save node embedding files for each timestamp |
| record_time | bool | whether or not to record the running time |

Note that **optional** parameters can be omitted in the configuration file if this optional parameter is not needed.  Moreover, the timestamp range is a closed interval \[start_idx, end_idx\]. The program will add 1 to `end_idx`, making the range into a left-closed interval. 

Usually, the `duration` parameter is greater than 1 for dynamic graph embedding; for static graph embedding, `duration` is set as 1.

### DynGEM parameters

Here we provide parameters used by DynGEM

| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: |
| beta | float |  reconstruction penalty |
| nu1 | float | relative weight hyper-parameter of L1-regularization loss |
| nu2 | float | relative weight hyper-parameter of L2-regularization loss |
| bias | bool | whether or not to enable bias for model layers |
| n_units | list |  hidden layer dimension list of its encoder |
| alpha | float | relative weight hyper-parameter of its first order proximity loss |

### GNN parameters
Here we provide input parameters of all supported GNN methods. 

#### CTGCN-C and CTGCN-S parameters
| **Parameter** | **Type** | **Description** |
|:----:|:----:| :----: | 
| nfeature_folder | str | node feature folder path relative to `base_path` <br> (**optional**, value: can be null) |
| learning_type | str | learning type of gnns <br> (value: 'U-neg', 'U-own', 'S-node', 'S-edge', 'S-link-st', 'S-link-dy') |
| hid_dim | int | dimension of hidden layers in a graph embedding model |
| weight_decay | int | weight decay of the optimizer |
| max_core | int | number of k-core subgraphs for each dynamic graph <br> (default: -1, means all k-core subgraphs in each graph are used) | Common  |
| trans_layer_num | int | feature transformation layer number | Common |
| diffusion_layer_num | int | core-based diffusion (or CGCN) layer number | Common |
| init_type | str | degree-based input feature initialization type <br> (**optional**, value: 'gaussian', 'one-hot', 'adj', 'combine')  | CTGCN-S |
| std | float | std of the gaussian distribution if `init_type` is 'gaussian' <br> (**optional**) | CTGCN-S |
| model_type | str | model type to identify different versions of the model <br> (value: 'C' or 'S', means C-version or S-version | Common |
| rnn_type | str | rnn type to identify different rnns used in CGCN <br> (value: 'GRU' or 'LSTM') | Common |
| trans_activate_type | str | activation function type of feature transformation layers <br> (value: 'L' or 'N', means 'linear' or 'non-linear') | Common |
| bias | bool | whether or not to enable bias for model layers | Common |

Note that `learning_type` is a parameter for choosing different learning strategies. Moreover, in all configuration files, we don't include `nfeature_folder` parameter, because all data sets used in this project don't have node features. But if you want to run programs on graph data sets with node features, you can add the `nfeature_folder` parameter in your configuration file.

Note that if `max_core` is greater than the k-core number of a graph, then all k-core subgraphs are used in CGCN layers.

#### Learning parameters
Learning parameters are parameters related to unsupervised learning or supervised learning. These parameters are all **optional parameters**. 

| **Parameter** | **Type** | **Description** | Usage |
|:----:|:----:| :----: | :----: |
| walk_pair_folder | str | random walk co-occurring node pairs folder path relative to `base_path` | Unsupervised Learning
| node_freq_folder | str | random walk node frequency folder path relative to `base_path` | Unsupervised Learning
| neg_num | int | negative sample number |  Unsupervised Learning |
| Q | float | penalty weight of negative sampling term in negative sampling loss| Unsupervised Learning |
| train_ratio | float | the ratio of training nodes(edges) to all nodes(edges) in each graph | Supervised Learning |
| val_ratio | float | the ratio of validation nodes(edges) to all nodes(edges) in each graph | Supervised Learning |
| test_ratio | float | the ratio of test nodes(edges) to all nodes(edges) in each graph | Supervised Learning |

Note that `walk_pair_folder`, `node_freq_folder`, `neg_num` and `Q` are used for unsupervised negative sampling loss. Parameters from `cls_file` to `cls_activate_type` are used for building the node classifier or edge classifier. `train_ratio`, `val_ratio` and `test_ratio` are used for data split in supervised node classification, edge classification and link prediction. 

Moreover, the input_dim of a classifier is `embed_dim`, and the output_dim of a classifier is the unique label number.

Here we introduce how to change different learning strategies.

- Unsupervised learning with negative sampling loss (`learning_type` = 'U-neg')
- Unsupervised learning with its own loss (`learning_type` = 'U-own')
