import numpy as np
import pandas as pd
import networkx as nx

if __name__ == "__main__":
    # Define the file paths
    path = '../result/'

    # Load edge data from CSV file (assuming the first row is a header)
    edges1 = np.loadtxt(path + 'gcase_m.csv', dtype=np.int32, delimiter=',', skiprows=1, usecols=[0, 1])
    nodes1 = np.loadtxt(path + 'node.csv', dtype=np.int32, delimiter=',', skiprows=1)

    # Create a new graph and add nodes and edges
    G1 = nx.Graph()
    G1.add_nodes_from(nodes1)
    G1.add_edges_from(edges1)

    # Calculate Eigenvector Centrality
    bc = nx.betweenness_centrality(G1)
    cc = nx.closeness_centrality(G1)
    dc = nx.degree_centrality(G1)
    ec = nx.eigenvector_centrality_numpy(G1)
    ks = nx.core_number(G1)
    # Create a list of centrality values for each node

    # Create a list of centrality values for each node
    bc_list = [{"Node": node, "Betweenness Centrality": bc[node]} for node in bc]
    cc_list = [{"Node": node, "Closeness Centrality": cc[node]} for node in cc]
    dc_list = [{"Node": node, "Degree Centrality": dc[node]} for node in dc]
    ec_list = [{"Node": node, "Eigenvector Centrality": ec[node]} for node in ec]
    ks_list = {"Node": list(ks.keys()), "K-Shell": list(ks.values())}
    # Create a DataFrame from the list
    bc = pd.DataFrame(bc_list)
    cc = pd.DataFrame(cc_list)
    dc = pd.DataFrame(dc_list)
    ec = pd.DataFrame(ec_list)
    ks = pd.DataFrame(ks_list)


    # Save the DataFrame to a CSV file
    bc.to_csv(path + 'bccontrol_m.csv', index=False)
    cc.to_csv(path + 'cccontrol_m.csv', index=False)
    dc.to_csv(path + 'dccontrol_m.csv', index=False)
    ec.to_csv(path + 'eccontrol_m.csv', index=False)
    ks.to_csv(path + 'kscontrol_m.csv', index=False)
