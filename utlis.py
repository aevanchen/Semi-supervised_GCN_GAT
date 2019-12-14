
import numpy as np
import networkx as nx
import scipy
from scipy import sparse
import torch
import torch.nn.functional as F
import warnings
import pickle as pkl
import sys
import os
import pandas as pd
import json
from scipy.sparse.linalg import inv,eigs
from train_utlis import get_train_test_set
warnings.filterwarnings("ignore")



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def get_degree(A):
    r, c = A.shape
    diags = A.sum(axis=1).flatten()
    D = scipy.sparse.spdiags(diags, [0], r, c, format='csr')
    return D


def get_laplacian(D, A):
    return D - A


def get_DH(A):
    # D^{-1/2}
    diags = A.sum(axis=1).flatten()
    C, R = A.shape
    with scipy.errstate(divide='ignore'):
        diag_s = 1.0 / scipy.sqrt(diags)
    diag_s[scipy.isinf(diag_s)] = 0
    DH = scipy.sparse.spdiags(diag_s, [0], C, R, format='csr')
    return DH


def get_spectrum_adj(A):
    Dh = get_DH(A)

    return sparse.csr_matrix(Dh.dot(A.dot(Dh)))



def read_json(file):
    with open(file) as json_data:
        data = json.load(json_data)
    return data


def get_graph(edges):
    try:
        edges['weight']
    except:
        edges['weight']=1
    result_edge = edges.values.tolist()
    G=nx.Graph()
    G.add_weighted_edges_from(result_edge)
    return G

def get_adjacency_matrix(graph):
    A =  nx.adjacency_matrix(graph)
    return A


def get_normalized_adjacency_matrix(graph,A):
    ind = range(len(graph.nodes()))
    inv_degs = [1/graph.degree(node) for node in graph.nodes()]
    inv_degs = sparse.csr_matrix(sparse.coo_matrix((inv_degs, (ind, ind)), shape=A.shape,dtype=np.float32))
    A = A.dot(inv_degs)
    return A


def get_laplacian(D,A):
    return D-A


#get normalized laplacian matrix 
# N = D^{-1/2} L D^{-1/2}
def get_normalized_laplacian(A):
    
    Dh=get_DH(A)
    D=get_degree(A)
    L=get_laplacian(D,A)
    return Dh.dot(L.dot(Dh)),Dh



def encode_onehot(labels):
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = 1 / rowsum.flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj(A):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    A_normalized = get_spectrum_adj(A + sparse.eye(A.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(A_normalized)



def load_github_data(data_dir='data/github_user'):
    edges = pd.read_csv(os.path.join(data_dir, 'musae_git_edges.csv'))
    edges=edges.sort_values(by=['id_1'])
    edges_dict=edges.groupby('id_1')['id_2'].apply(list).to_dict()
    graph=nx.from_dict_of_lists(edges_dict, create_using=nx.Graph())
    f =read_json(os.path.join(data_dir, 'musae_git_features.json'))
    features=np.zeros((len(f.keys()),4005))
    for i in f.keys():
        for j in f[i]:
                features[int(i)][j]=1
    labels=load_github_label(graph,data_dir='data/github_user')
    
    full_idx=np.arange(len(features)).reshape(-1,1)
    labels=torch.LongTensor(labels)
    features=torch.FloatTensor(features)
    label_lists=labels.numpy()
    x_train, x_test, y_train, y_test =get_train_test_set(full_idx, label_lists, train_test_split_coef=0.2)
    idx_train = torch.LongTensor(x_train.reshape(-1))
    idx_test = torch.LongTensor(x_test.reshape(-1))
    A = nx.adjacency_matrix(graph)
    A_processed = preprocess_adj(A)
    
    return (graph,A_processed,features,labels,idx_train,idx_test,idx_test)

def load_github_label(graph,data_dir='data/github_user'):
    target = pd.read_csv(os.path.join(data_dir, 'musae_git_target.csv'))
    label_dict=target.set_index('id').to_dict('index')
    label=np.array([label_dict[node]['ml_target'] for node in graph.nodes()])
    return label
                      
                      
def load_karate_club(data_dir='data/karate'):

    def set_atrribute(data,attributes):
        for col in attributes.columns.values:
                nx.set_node_attributes(
                    data,
                    values=pd.Series(
                        attributes[col],
                        index=attributes.index).to_dict(),
                    name=col
                )
        
    
    data = nx.read_edgelist(
        os.path.join(data_dir, 'karate.edgelist'),create_using=nx.Graph(),
        nodetype=int)

    attributes = pd.read_csv(
        os.path.join(data_dir, 'karate.attributes.csv'),
        index_col=['node'])
    
    set_atrribute(data,attributes)

    return data



def load_gcn_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str,dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G=nx.from_dict_of_lists(graph, create_using=nx.Graph())
    A = nx.adjacency_matrix(G)

    labels = np.vstack((ally, ty))

    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_test = test_idx_range.tolist()

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    features = normalize_feature(features)

    features = torch.FloatTensor(np.array(features.todense()))

    if dataset_str == 'citeseer':
        kk = np.zeros(len(labels)).astype(int)
        for i in range(len(labels)):
            t = labels[i]
            if sum(t) == 0:
                kk[i] = len(t)
            else:
                kk[i] = np.argwhere(t != 0)[0]
        labels=kk
        labels = torch.LongTensor(labels)
    else:
        labels = torch.LongTensor(np.where(labels)[1])




    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)

    A_processed = preprocess_adj(A)
    return (G,A_processed, features, labels, idx_train, idx_test, idx_val)


def load_data(data_name):
    if data_name=='github_user':
        data=load_github_data()
 
    elif data_name=='karate2':
        graph=load_karate_club()
        node_list,label_list =list(zip(*[ [node,label['community'] ]for node, label in sorted(graph.nodes(data=True))]))
        label= np.array([0 if l=='Administrator' else 1 for l in label_list ])
        features=get_adjacency_matrix(graph).A
        data=(graph,features,label)
    elif data_name=='karate4':
        graph=load_karate_club()
        node_list,label_name =list(zip(*[ [node,label['new_community'] ]for node, label in sorted(graph.nodes(data=True))]))
        label=np.array(label_name)
        features=get_adjacency_matrix(graph).A
        data=(graph,features,label)
    else:
        data=load_gcn_data(data_name)
    return data
                      
def get_spectrum_embedding(G,k): 
    # dimension of embedding size
    A = get_adjacency_matrix(G)

    D=get_degree(A)
    A_normalized=get_normalized_adjacency_matrix(G,A)

    L=get_laplacian(D,A)

    Lnorm,Dh=get_normalized_laplacian(A)

    E, V = eigs(Lnorm,k+1,which='SR')
    E = np.real(E)
    V = np.real(V)

    Z = Dh@V[:,1:k+1]*1000
    return Z



