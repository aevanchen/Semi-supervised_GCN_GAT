
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
from utlis import load_data, loss, accuracy
from tqdm import tqdm,tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold



DATASET=['cora','citeseer','pubmed']
DEPTH_LIST=[1,2,5,10,15,50,100,200,500,1000]
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
    A_norm, features, labels, idx_train,idx_test,idx_val=load_data(dataset_str)
        
   
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
    
    
    nfolds=5
    kf = KFold(n_splits=nfolds)
    index_list=np.arange(len(features))
    total_train_acc=0
    total_test_acc=0
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

            loss_train,acc_train=train(model,idx_train)

            if not fast_training:

                loss_val,acc_val=evaluate(model,idx_test)

                if loss_val<min_val_loss:
                  epoch_no_improvement=0
                  min_val_loss = loss_val
                else:
                  epoch_no_improvement+=1

                if epoch_no_improvement==early_stop_epoch:

                  break
        
        loss_train,acc_train=evaluate(model,idx_train)
        loss_val,acc_val=evaluate(model,idx_test)
        total_train_acc+=acc_train
        total_test_acc+=acc_val
        t+=1
                
    return total_train_acc/t,total_test_acc/t




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
        axes[i].plot(x, train_with_res_temp,'g-', label="Train (Residual)",marker='o')
        axes[i].plot(x, test_with_res_temp,'r-', label="Test (Residual)", marker='o')
        axes[i].plot(x, train_without_res_temp,'b--', label="Train", marker='o')
        axes[i].plot(x, test_without_res_temp,'y--',label="Test", marker='o')

        axes[i].set_xlabel("Number of Layers")
        axes[i].set_ylabel("Accuracy")
        axes[i].title.set_text(dataset_str)
        axes[i].legend(loc=4)
        axes[i].set_xticks(range(1,len(train_with_res_temp)+1))


        train_with_res.append(train_with_res_temp)
        test_with_res.append(test_with_res_temp)
        train_without_res.append(train_without_res_temp)
        test_without_res.append(test_without_res_temp)
    plt.savefig("depth_cross.jpg")
    return 

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

def plot_runtime_analysis(ndepth_list=DEPTH_LIST):
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

def main():
    parser = argparse.ArgumentParser(description='Set up')
    parser.add_argument('--seed', type=int, default=27, help='Random seed.')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay.')
    parser.add_argument('--plot_depth', action='store_true', default=False,
                        help='plot_depth')
    parser.add_argument('--plot_time', action='store_true', default=False,
                        help='plot_time')
  

    args = parser.parse_args()
    args.cuda=torch.cuda.is_available() and not args.no_cuda
    args.device =torch.device("cuda" if args.cuda else "cpu")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    if args.plot_depth:
        plot_depth_analysis()
        
    if args.plot_time:
        plot_runtime_analysis()
        
        
    
    
    return 1
        


if __name__ == '__main__':
     
    
    
    main()
