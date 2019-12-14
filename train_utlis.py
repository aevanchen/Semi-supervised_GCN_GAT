
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
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,read_edgelist, set_node_attributes
from sklearn.metrics import average_precision_score,recall_score,precision_score,accuracy_score
from sklearn.manifold import TSNE
                
def loss(output, labels, idx):
    output = F.log_softmax(output[idx])
    loss = F.nll_loss(output, labels[idx])
    return loss


def accuracy(output, labels, idx):
    output = F.log_softmax(output[idx])
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels[idx]).double()
    correct = correct.sum()
    return correct / len(labels[idx])


  
def train(idx_train):
  
    model.train()
   
    output = model(features)
    
    loss_train=loss(output,labels,idx_train)
    acc_train = accuracy(output, labels,idx_train)
    optimizer.zero_grad()
    loss_train.backward()
    
    optimizer.step()
   
    return loss_train.item(),acc_train.item()


def evaluate(idx):
    model.eval()
    output = model(features)
    loss_ =loss(output,labels,idx)
    acc_ = accuracy(output, labels,idx)

    return loss_.item(),acc_.item()



def get_train_test_set(x, label_lists,train_test_split_coef=0.8):
    n_labels=len(list(set(label_lists)))
    x_train_index=[]
    x_test_index=[]
    for i in range(n_labels):
        a=np.where(label_lists==i)[0]
        x_train_index=x_train_index+list(a[:int(len(a)*train_test_split_coef)])
        x_test_index=x_test_index+list(a[int(len(a)*train_test_split_coef):])
    x_train=x[x_train_index,:]
    x_test=x[x_test_index,:]
    y_train=label_lists[x_train_index]
    y_test=label_lists[x_test_index]

    return (x_train, x_test, y_train, y_test)



def fit_predict_lr(x_train, x_test, y_train, y_test):
    n_labels=len(list(set(y_train)))
    if n_labels==2:
        avg='binary'
    else:
        avg='weighted'
    def prediction(clf,x,n_labels):
        if n_labels==2:
            pred=clf.predict(x)
            pred[pred>=0.5]=1
            pred[pred<0.5]=0
        else:
            pred=clf.predict(x)
        return pred
        
    clf = LogisticRegression(solver='liblinear',max_iter=500).fit(x_train, y_train)

    train_pred=prediction(clf ,x_train,n_labels)
    test_pred=prediction(clf ,x_test,n_labels)

    print("Train acc: ",accuracy_score(y_train, train_pred) )
    
    print("Training precision: ",precision_score(y_train,train_pred,average=avg),'Training recall: ',recall_score(y_train ,train_pred,average=avg) )
    
    print("Test acc: ",accuracy_score(y_test, test_pred) )
    
    print("Testing precision: ",precision_score(y_test,test_pred,average=avg),'Training recall: ',recall_score(y_test ,test_pred,average=avg) )
    
    return accuracy_score(y_train, train_pred),accuracy_score(y_test, test_pred) 
    
def get_tsne_results(features,n_components=2):
    tsne = TSNE(n_components=n_components, verbose=0, perplexity= 25, n_iter=500)
    tsne_results=tsne.fit_transform(features)
    return tsne_results