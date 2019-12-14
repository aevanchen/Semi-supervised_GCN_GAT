import argparse
import numpy as np
import networkx as nx
import scipy
from scipy import sparse
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
import time
from models import SimpleGCN,MutipleGCN
from utlis import load_data
from tqdm import tqdm,tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from train_utlis import *
from plot_utlis import *
import pandas as pd
import json
import numpy as np
import networkx as nx
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length,read_edgelist, set_node_attributes
from sklearn.metrics import average_precision_score,recall_score,precision_score,accuracy_score
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
import time
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.sparse.linalg import inv,eigs
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.sparse import csc_matrix
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")

DATASET=['cora','citeseer','pubmed']
DEPTH_LIST=[1,2,3,4,5,6,7,8,9,10]
no_cuda= False,
seed=42
epochs=400
lr=0.01
weight_decay=5e-4
hidden=16
dropout=0.5
fast_training=False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
early_stop_epoch=10



def train_model_depth(dataset_str,depth=None,res_connection=False):
    G,A_norm, features, labels, idx_train,idx_test,idx_val=load_data(dataset_str)
        
   
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
    
    
    nfolds=5
    kf = KFold(n_splits=nfolds)
    index_list=np.arange(len(features))
    total_train_acc=[]
    total_test_acc=[]
    t=0
    for idx_train, idx_test in kf.split(index_list):
        model = MutipleGCN(adj=A_norm,ngcu=depth,nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout,res_connection=res_connection).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        
        idx_train=torch.LongTensor(idx_train)
        idx_test=torch.LongTensor(idx_test)
    # Train model
        t_total = time.time()
        epoch_no_improvement=0
        min_val_loss=float('inf')
        for epoch in tqdm(range(epochs)):

            loss_train,acc_train=train(idx_train)
        
        loss_train,acc_train=evaluate(idx_train)
        loss_val,acc_val=evaluate(idx_test)
        
        t+=1
        total_train_acc.append(acc_train)
        total_test_acc.append(acc_val) 
    return total_train_acc,total_test_acc




def train_model_time(dataset_str,depth=None,res_connection=False):
    A_norm, features, labels, _,_,_=load_data(dataset_str)
    
    idx_train=np.arange(len(features))
    idx_train=torch.LongTensor(idx_train)
    
    def train(model,idx_train): 
        model.train()

        output = model(features)

        loss_train=loss(output,labels,idx_train)
        acc_train = accuracy(output, labels,idx_train)
        optimizer.zero_grad()
        loss_train.backward()

        optimizer.step()

        return loss_train.item(),acc_train.item()
    def evaluate(model,idx):
        model.eval()
        output = model(features)
        loss_ =loss(output,labels,idx)
        acc_ = accuracy(output, labels,idx)

        return loss_.item(),acc_.item()
    

    model = MutipleGCN(adj=A_norm,ngcu=depth,nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout,res_connection=res_connection).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    t_start = time.time()
    for i in range(10):
        loss_train,acc_train=train(model,idx_train)
            
    training_time=(time.time() - t_start)/10
    
    t_start = time.time()
    for i in range(10):
        loss_val,acc_val=evaluate(model,idx_train)
    
    inference_time=(time.time() - t_start)/10
    
    return training_time,inference_time



def plot_depth_analysis(ndepth=10):
    plt.figure(figsize=(16,6))
    ax1=plt.subplot(131)
    ax2=plt.subplot(132)
    ax3=plt.subplot(133)
    axes=[ax1,ax2,ax3]
    train_with_res=[]
    test_with_res=[]
    train_without_res=[]
    test_without_res=[]
    ndepth=ndepth
    for i,dataset_str in enumerate(DATASET):

        print(dataset_str)
        train_with_res_temp=[]
        test_with_res_temp=[]
        train_without_res_temp=[]
        test_without_res_temp=[]
        for depth in range(1,ndepth+1):

            trn1,tst1=train_model_depth(dataset_str,depth=depth,res_connection=True)

            trn2,tst2=train_model_depth(dataset_str,depth=depth,res_connection=False)
            
            train_with_res_temp.append(trn1)
            test_with_res_temp.append(tst1)
            train_without_res_temp.append(trn2)
            test_without_res_temp.append(tst2)

        
        x=range(1,len(train_with_res_temp)+1)
        
        train_with_res_temp=np.array(train_with_res_temp).T
        test_with_res_temp=np.array(test_with_res_temp).T
        train_without_res_temp=np.array(train_without_res_temp).T
        test_without_res_temp=np.array(test_without_res_temp).T
        
        var_1_min=np.min(train_with_res_temp,axis=0)
        var_1_mean=np.mean(train_with_res_temp,axis=0)
        var_1_max=np.max(train_with_res_temp,axis=0)
        
        var_2_min=np.min(test_with_res_temp,axis=0)
        var_2_mean=np.mean(test_with_res_temp,axis=0)
        var_2_max=np.max(test_with_res_temp,axis=0)
        
        var_3_min=np.min(train_without_res_temp,axis=0)
        var_3_mean=np.mean(train_without_res_temp,axis=0)
        var_3_max=np.max(train_without_res_temp,axis=0)
        
        var_4_min=np.min(test_without_res_temp,axis=0)
        var_4_mean=np.mean(test_without_res_temp,axis=0)
        var_4_max=np.max(test_without_res_temp,axis=0)
        
        x=range(1,len(var_1_mean)+1)
        axes[i].plot(x, var_1_mean,'g-', label="Train (Residual)",marker='o')
        axes[i].plot(x, var_2_mean,'r--', label="Test (Residual)", marker='o')
        axes[i].plot(x, var_3_mean,'p-', label="Train", marker='o')
        axes[i].plot(x, var_4_mean,'b--',label="Test", marker='o')
        axes[i].fill_between(x, var_1_min, var_1_max,
                 color='green', alpha=0.1)
        axes[i].fill_between(x, var_2_min, var_2_max,
                 color='red', alpha=0.1)
        axes[i].fill_between(x, var_3_min, var_3_max,
                 color='purple', alpha=0.1)
        axes[i].fill_between(x, var_4_min, var_4_max,
                 color='blue', alpha=0.1)

        axes[i].set_xlabel("Number of Layers")
        axes[i].set_ylabel("Accuracy")
        axes[i].title.set_text(dataset_str)
        axes[i].legend(loc=3)
        axes[i].set_xticks(range(1,len(var_1_mean)+1))


        train_with_res.append(train_with_res_temp)
        test_with_res.append(test_with_res_temp)
        train_without_res.append(train_without_res_temp)
        test_without_res.append(test_without_res_temp)
    plt.savefig("depth_cross_new.jpg")
    #plt.show()
    return
#tsne scatter plot
def scatter_plot(x,label_lists,data_set):
    df=pd.DataFrame()
    df['y']=label_lists
    df['1d']=x[:,0]
    df['2d']=x[:,1]
   
    fig = go.Figure( layout=go.Layout(
            xaxis=dict(showgrid=False, zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False,showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
        ))


    for i in range( len(set(label_lists))):
        fig.add_trace(go.Scatter(
            x=df[df['y']==i]['1d'].to_numpy(),
            y=df[df['y']==i]['2d'].to_numpy(),
            #z= df[df['y']==i]['3d'].to_numpy(),
            mode='markers',
            name="Label{}".format(str(i+1)),
            marker=dict(
                size=8,
                symbol='circle',
                color=i,
                opacity=1
            )
        ))

    fig.update_layout(
        title= {'text':data_set,
        'font': {
          'family': 'Courier New, monospace',
          'size': 30
        },
        'xref': 'paper',
        'x': 0.5},
        xaxis_title="TSNE-1d",
        yaxis_title="TSNE-2d",
        font=dict(
            family="Courier New, monospace",
            size=11,
            color="#7f7f7f"
        )

    )
    fig.show()
      
    return fig



def plot_runtime_analysis(ndepth_list):
    plt.figure(figsize=(10,6))
    ax1=plt.subplot(111)


    dataset_str='cora'
    train_with_res_temp=[]
    test_with_res_temp=[]
    test_without_res_temp=[]
    train_without_res_temp=[]
    for depth in ndepth_list:

        trn_time1,tst_time1=train_model_time(dataset_str,depth=depth,res_connection=True)
        
        trn_time2,tst_time2=train_model_time(dataset_str,depth=depth,res_connection=False)
        
        train_with_res_temp.append(trn_time1)
        test_with_res_temp.append(tst_time1)
        train_without_res_temp.append(trn_time2)
        test_without_res_temp.append(tst_time2)


    x=range(1,len(ndepth_list)+1)
    ax1.plot(x, train_with_res_temp,'g-', label="Training (Residual)",marker='o')
    ax1.plot(x, test_with_res_temp,'r-', label="Inference (Residual)", marker='o')
    ax1.plot(x, train_without_res_temp,'b--', label="Training", marker='o')
    ax1.plot(x, test_without_res_temp,'y--',label="Inference", marker='o')

    ax1.set_xlabel("Number of Layers")
    ax1.set_ylabel("Time")
    ax1.title.set_text("Running time analysis of GCN")
    ax1.legend(loc=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ndepth_list)
    

    plt.savefig("run_time_plot.jpg")
    return 