import networkx as nx


dataset = 'Amazon_setting1'


def count_lines(file_path):
    with open(file_path, 'r') as file:
        num_lines = sum(1 for _ in file)
    return num_lines


num_nodes = []
num_edges = []
num_communities = []
mean_degree = []
modularity = []
biggest_community = []
smallest_community = []
mean_community = []
for t in range(10):
    num_nodes.append(count_lines(f'./data/{dataset}/3.label/communities{t}.txt'))
    num_edges.append(count_lines(f'./data/{dataset}/1.format/graph{t}.txt'))
    num_communities.append(count_lines(f'../DynCLARE/dataset/{dataset}/time_{t}/{dataset}_{t}-1.90.cmty.txt'))
    mean_degree.append(num_edges[-1]/num_nodes[-1])
    graph_path = f'./data/{dataset}/1.format/graph{t}.txt'
    node_path = f'./data/{dataset}/3.label/communities{t}.txt'
    # Create the list of nodes
    node_list = []
    with open(node_path, 'r') as file:
        for line in file:
            node, community = line.split()
            node_list.append(int(node))
    # Create the graph
    with open(graph_path, 'r') as file:
        g = nx.Graph()
        g.add_nodes_from(node_list)  # Verificare come funge
        for line in file:
            node1, node2 = map(int, line.strip().split())
            # edges = [[mapping[u], mapping[v]] for u, v in edges]
            g.add_edge(node1, node2, weight=1)  # Devo fare il mapping
    # Calculate modularity
    communities = []
    biggest_community_size = 0
    smallest_community_size = 1000000
    all_sizes = []
    with open(f'../DynCLARE/dataset/{dataset}/time_{t}/{dataset}_{t}-1.90.cmty.txt', 'r') as file:
        for line in file:
            size = len(line.split())
            all_sizes.append(size)
            if size > biggest_community_size:
                biggest_community_size = size
            if size < smallest_community_size:
                smallest_community_size = size
            communities.append(set(map(int, line.split())))
    # print('Num nodes for the communities ', sum(all_sizes))
    # print('Num nodes for the graph ', g.number_of_nodes())
    modularity.append(nx.algorithms.community.quality.modularity(g, communities))
    biggest_community.append(biggest_community_size)
    smallest_community.append(smallest_community_size)
    mean_community.append(sum(all_sizes)/len(all_sizes))

#Save to file the results
with open(f'./data/{dataset}/characteristics.txt', 'w') as file:
    file.write('CHARACTERISTICS FOR EACH TIMESTEP\n')
    file.write('Number of nodes: ' + str(num_nodes) + '\n')
    file.write('Number of edges: ' + str(num_edges) + '\n')
    file.write('Number of communities: ' + str(num_communities) + '\n')
    file.write('Mean degree: ' + str(mean_degree) + '\n')
    file.write('Modularity: ' + str(modularity) + '\n')
    file.write('Biggest community size: ' + str(biggest_community) + '\n')
    file.write('Smallest community size: ' + str(smallest_community) + '\n')
    file.write('Average community size: ' + str(mean_community) + '\n')
    file.write('MEAN CHARACTERISTICS\n')
    file.write('Mean number of nodes: ' + str(sum(num_nodes)/len(num_nodes)) + '\n')
    file.write('Mean number of edges: ' + str(sum(num_edges)/len(num_edges)) + '\n')
    file.write('Mean number of communities: ' + str(sum(num_communities)/len(num_communities)) + '\n')
    file.write('Mean degree: ' + str(sum(mean_degree)/len(mean_degree)) + '\n')
    file.write('Mean modularity: ' + str(sum(modularity)/len(modularity)) + '\n')
    file.write('Mean biggest community size: ' + str(sum(biggest_community)/len(biggest_community)) + '\n')
    file.write('Mean smallest community size: ' + str(sum(smallest_community)/len(smallest_community)) + '\n')
    file.write('Mean average community size: ' + str(sum(mean_community)/len(mean_community)) + '\n')

# Print the results
print('CHARACTERISTICS FOR EACH TIMESTEP')
print('Number of nodes: ', num_nodes)
print('Number of edges: ', num_edges)
print('Number of communities: ', num_communities)
print('Average degree: ', mean_degree)
print('Modularity: ', modularity)
print('Biggest community size: ', biggest_community)
print('Smallest community size: ', smallest_community)
print('Average community size: ', mean_community)
print('MEAN CHARACTERISTICS')
print('Mean number of nodes: ', sum(num_nodes)/len(num_nodes))
print('Mean number of edges: ', sum(num_edges)/len(num_edges))
print('Mean number of communities: ', sum(num_communities)/len(num_communities))
print('Mean average degree: ', sum(mean_degree)/len(mean_degree))
print('Mean modularity: ', sum(modularity)/len(modularity))
print('Mean biggest community size: ', sum(biggest_community)/len(biggest_community))
print('Mean smallest community size: ', sum(smallest_community)/len(smallest_community))
print('Mean average community size: ', sum(mean_community)/len(mean_community))





