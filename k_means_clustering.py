from evaluation import community_detection as cd
from utils import mean_n_std
import os


dataset = 'Amazon_setting1'
method = 'DynGEM'
train_ratio = 0.5
pred_community_path = f'./data/{dataset}/pred_community/{method}'
score_path = f'./data/{dataset}/community_detection_score/{method}'

if not os.path.exists(pred_community_path):
    os.makedirs(pred_community_path)
if not os.path.exists(score_path):
    os.makedirs(score_path)

for i in range(10):
    print(f'STARTING TIMESTEP {i}')
    train_dict, test_dict, embeddings, active_nodes, dataloader, emb_dim = cd.load_data(dataset, method, i, train_ratio)
    cd.evaluate_community_detection(train_dict, test_dict, embeddings, active_nodes, pred_community_path, score_path, i, emb_dim, dataloader, False)

mean_n_std(score_path + '/score_kmeans.txt')