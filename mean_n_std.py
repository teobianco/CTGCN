import numpy as np

dataset = 'Amazon'
setting = 'setting3'
method = 'DynGEM'


def mean_n_std(dataset, setting, method):
    score_file_path = f'./data/{dataset}_{setting}/community_detection_score/{method}/score.txt'
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


mean_n_std(dataset, setting, method)

