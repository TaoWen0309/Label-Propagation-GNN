import networkx as nx
import numpy as np
import gudhi as gd
import torch
import random

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx

def sublevel_persistence_diagram(A, method, max_scale=50):
    assert method in ['degree','betweenness','eigenvector','closeness']
    
    G = nx.from_numpy_array(A)
    if method == 'degree':
        node_features = np.sum(A, axis=1)
    elif method == 'betweenness':
        node_features_dict = nx.betweenness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'eigenvector':
        node_features_dict = nx.eigenvector_centrality(G,max_iter=100000)
        node_features = [i for i in node_features_dict.values()]
    elif method == 'closeness':
        node_features_dict = nx.closeness_centrality(G)
        node_features = [i for i in node_features_dict.values()]
    
    stb = gd.SimplexTree()
    (xs, ys) = np.where(np.triu(A))
    for j in range(A.shape[0]):
        stb.insert([j], filtration=-1e10)

    for idx, x in enumerate(xs):
        stb.insert([x, ys[idx]], filtration=-1e10)

    for j in range(A.shape[0]):
        stb.assign_filtration([j], node_features[j])

    stb.make_filtration_non_decreasing()
    dgm = stb.persistence()
    pd = [dgm[i][1] if dgm[i][1][1] != np.inf else (dgm[i][1][0], max_scale) for i in np.arange(0, len(dgm), 1)]

    return np.array(pd)

def persistence_images(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1.):
    PXs, PYs = dgm[:, 0], dgm[:, 1]
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    max_output = np.max(output)
    min_output = np.min(output)
    if normalization and (max_output != min_output):
        norm_output = (output - min_output)/(max_output - min_output)
    else:
        norm_output = output

    return norm_output


def compute_PI_tensor(graph_list,PI_dim,sublevel_filtration_methods=['degree','betweenness','eigenvector','closeness']):
    PI_list = []
    for graph in graph_list:
        adj = nx.adjacency_matrix(to_networkx(graph)).todense()
        PI_list_i = [] 
        # PI tensor
        for j in range(len(sublevel_filtration_methods)):
            pd = sublevel_persistence_diagram(adj,sublevel_filtration_methods[j])
            pi = torch.FloatTensor(persistence_images(pd,resolution=[PI_dim]*2))
            PI_list_i.append(pi)
        PI_tensor_i = torch.stack(PI_list_i)
        PI_list.append(PI_tensor_i)

    PI_concat = torch.stack(PI_list)
    return PI_concat

def target_source_sim(pseudo_labels, labels_source, output_source):
    sim_dict = {}
    for label in pseudo_labels.unique():
        indices = (labels_source == label).nonzero(as_tuple=True)[0]
        sim_dict[label.item()] = torch.mean(output_source[indices], dim=0)
    return sim_dict

def label_propagation(output_source, output_target, labels_source, threshold, criterion, device):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    max_probs, pseudo_labels = output_target.max(1)
    mask = max_probs.ge(threshold) # sufficiently confident
    if torch.any(mask):
        pseudo_labels = pseudo_labels[mask]
        sim_output = target_source_sim(pseudo_labels, labels_source, output_source)
        for label in pseudo_labels:
            loss = criterion(sim_output[label.item()], label)
            total_loss += loss
    return total_loss/len(pseudo_labels)

def reg_scheduler(reg1, reg2, iter_num, epoch, total_iters, gamma=0.01, power=0.75):
    reg1 *= (1 + gamma * (epoch * total_iters + iter_num)) ** (-power)
    reg2 *= (1 + gamma * (epoch * total_iters + iter_num)) ** (-power)
    return reg1, reg2