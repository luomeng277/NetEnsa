import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def input_connectable(node_list, adjacency_dict):
    reachable_nodes = set()
    for node in node_list:
        reachable_nodes.update(find_reachable_nodes(node, adjacency_dict))
    return reachable_nodes, set(node_list)

def sc(adjacency_dict, B):
    for i in range(len(B)):
        if not set(B[i]).issubset(adjacency_dict[i]):
            return 0
    return 1


def ssc(As, Bs):

    combined_matrix = np.hstack((As, Bs))

    # Check the rank condition
    if np.linalg.matrix_rank(combined_matrix) == As.shape[0]:
        return 1
    else:
        return 0

def find_reachable_nodes(start_node, adjacency_dict):
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adjacency_dict[node] - visited)

    return visited

def find_indices_matching(all_n, sel_n):
    matching_indices = []
    for i in range(sel_n.shape[0]):
        for k in range(all_n.shape[0]):
            if all_n.iloc[k, 0] == sel_n.iloc[i, 0]:
                matching_indices.append(k)
    return matching_indices

def remove_isolated_nodes(graph):
    isolated_nodes = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated_nodes)
    return graph

if __name__ == "__main__":
    path= '../result/'

    edges = np.loadtxt(path + 'gcase_m.csv', dtype=np.int32, delimiter=',', skiprows=1, usecols=[0, 1])
    nodes = np.loadtxt(path + 'node.csv', dtype=np.int32, delimiter=',', skiprows=1)
    sel_n = pd.read_csv(path+ 'aden.csv', skiprows=0)
    all_n = pd.read_csv(path + 'alldata.csv', index_col=0).T.columns

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)  # You can choose a different layout
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, font_color='black', edge_color='gray', width=1, alpha=0.7)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    # Show the plot (optional)
    plt.show()
    components = list(nx.connected_components(G))
    isolated_nodes = [node for node_set in components if len(node_set) == 1 for node in node_set]
    num_isolated_nodes = len(isolated_nodes)


    print(f'have {num_isolated_nodes}  no Neighbourhood nodes')


    if num_isolated_nodes > 0:

        num_nodes_before = len(G.nodes)
        num_edges_before = len(G.edges)

        G.remove_nodes_from(isolated_nodes)
        print(f'delect {num_isolated_nodes} isolated nodes')

        num_nodes_after = len(G.nodes)
        num_edges_after = len(G.edges)
        print(f"Nodes before: {num_nodes_before}, Edges before: {num_edges_before}")
        print(f"Nodes after: {num_nodes_after}, Edges after: {num_edges_after}")
        components_after = list(nx.connected_components(G))
        for i, component in enumerate(components_after):
            print(f"Component {i + 1}: {(component)}")
            print(f"Component {i + 1}: {len(component)} nodes")

    G = remove_isolated_nodes(G)
    print("Nodes before:", len(G.nodes), "Edges before:", len(G.edges))

    S = coo_matrix((np.ones(len(edges)), (np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))))
    An = S.tocsr().transpose().toarray()
    G = nx.Graph()
    for i, row in enumerate(An):
        G.add_node(i)
        G.add_edges_from((i, j) for j in np.nonzero(row)[0])


    L = find_indices_matching(all_n, sel_n)

    D_l, Mc = input_connectable(set(L), G)


    print(f"Intersection: {set(L) & set(G.nodes)}")

    B = np.zeros_like(An)

    for i in range(len(An)):
        for j in range(len(L)):
            if i == L[j] and i < len(B): 
                B[i, j] = 1

    cond1 = sc(An, B)

    if cond1 == 1:
        print('Network control conforms to SCC')
    else:
        print('Network control conforms to SC')
