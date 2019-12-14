import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    GCN layer
    """

    def __init__(self, adj, input_dims, output_dims ,res_connection=False):
        super(GCNLayer, self).__init__()
        self.in_dims= input_dims
        self.out_dims = output_dims
        self.linear =nn.Linear(self.in_dims, self.out_dims, bias=True)

        self.res = nn.Linear(self.in_dims, self.out_dims, bias=True)
        self.adj =adj
        self.res_con =res_connection
        self.init_weights([self.linear ,self.res])
        self.relu = nn.ReLU()

    def forward(self, input_feature,last_hidden=None):


        output = torch.spmm(self.adj, input_feature)
        output = self.linear(output)
        if  self.res_con and last_hidden is not None:
            if last_hidden.size(1)!=output.size(1):
                output =self.relu(output) + self.res(last_hidden)
            
            else:
                output =self.relu(output) + last_hidden

        return output

    def init_weights(self,layers):
        # xavier weights initialization
        for layer in layers:
            torch.nn.init.xavier_uniform(layer.weight)


class SimpleGCN(nn.Module):
    def __init__(self, adj, nfeat, nhid, nclass, dropout, res_connection=False,get_hidden=False):
        super(SimpleGCN, self).__init__()
        self.adj = adj
        self.gc1 = GCNLayer(self.adj, nfeat, nhid, res_connection=res_connection)
        self.gc2 = GCNLayer(self.adj, nhid, nclass, res_connection=res_connection)
        self.dropout_rate = dropout
        self.get_hidden=get_hidden


    def forward(self, x,last_hidden=None):
        last_hidden=None
        x = self.gc1(x)
        last_hidden=x
        if self.get_hidden:
            hidden=x
        x = F.dropout(x, self.dropout_rate, self.training)
        x = self.gc2(x,last_hidden)
        
        if self.get_hidden:
            return x,hidden
        return x

class MutipleGCN(nn.Module):
    def __init__(self, adj, ngcu, nfeat, nhid, nclass, dropout, res_connection=False):
        super(MutipleGCN, self).__init__()
        self.adj = adj
        self.ngcu = ngcu
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.res_connection=res_connection
        self.dropout_rate = dropout
        self.layers= nn.ModuleList(self.init_layers(self.ngcu))
        self.get_hidden=False


    def forward(self, x):
        if len(self.layers) == 1:

            x = self.layers[0](x)
        else:
            last_hidden=None
            for layer in self.layers[:-1]:
                x = layer(x,last_hidden)
                last_hidden=x
                x = F.dropout(x, self.dropout_rate, self.training)
                
            x = self.layers[-1](x)
        return x

    def init_layers(self, ngcu):
        if ngcu == 1:
            return [GCNLayer(self.adj, self.nfeat, self.nclass, res_connection=self.res_connection)]
        elif ngcu == 2:
            return [GCNLayer(self.adj, self.nfeat, self.nhid, res_connection=self.res_connection),GCNLayer(self.adj, self.nhid, self.nclass, res_connection=self.res_connection)]
        else:
            res = [GCNLayer(self.adj, self.nfeat, self.nhid, res_connection=self.res_connection)]
            for i in range(1,ngcu-1):
                res += [GCNLayer(self.adj, self.nhid, self.nhid, res_connection=self.res_connection)]
            res += [GCNLayer(self.adj, self.nhid, self.nclass, res_connection=self.res_connection)]
            return res



class GAT(nn.Module):
    def __init__(self, adj,nheads,nfeat, nhid, nclass, dropout=0.6,res_connection=False,last_layer_mh=False):
   
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj=adj
        self.nfeat=nfeat
        self.nhid=nhid
        self.nheads=nheads
        self.last_layer_mh=last_layer_mh
        self.attentions =  nn.ModuleList([GraphAttentionLayer(self.adj,nfeat, 
                                                 nhid, 
                                                 dropout=self.dropout, 
                                                 apply_act=True,res_connection=res_connection) for _ in range(self.nheads)])
        
        if not self.last_layer_mh:
            self.out = GraphAttentionLayer(self.adj,nhid * nheads, 
                                                 nclass, 
                                                 dropout=self.dropout,
                                                 apply_act=False)
        else:
            self.out = nn.ModuleList([GraphAttentionLayer(self.adj,nhid * nheads, 
                                                 nclass, 
                                                 dropout=0, 
                                                 apply_act=False,res_connection=False) for _ in range(2)])
            
     

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att_layer(x) for att_layer in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        
        if not  self.last_layer_mh:
            x = self.out(x)
        else:
            l=torch.stack([att_layer(x) for att_layer in self.out])
            
            x=torch.mean(torch.stack([att_layer(x) for att_layer in self.out]),axis=0)

        return x
    

class SpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    """Rerefence: https://github.com/Diego999/pyGAT/blob/master/layers.py"""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class Spmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpmmFunction.apply(indices, values, shape, b)

    
class GraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer
    """
    def __init__(self,adj, in_dims, out_dims, dropout, apply_act=True,res_connection=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.apply_act = apply_act
        
        self.adj=adj
        
        self.dropout =dropout
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.special_spmm = Spmm()
       
        self.W = nn.Parameter(torch.zeros(size=(self.in_dims, self.out_dims)))
        nn.init.xavier_uniform(self.W.data)
        #nn.Linear(self.in_dims, self.out_dims, bias=True)
        
        self.res = nn.Parameter(torch.zeros(size=(self.in_dims, self.out_dims)))
        nn.init.xavier_uniform(self.res.data)
        
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_dims)))
        nn.init.xavier_uniform(self.a.data)
     
        self.res_connection=res_connection
    
    def forward(self, x):
        n = x.size()[0]
        
        indices = self.adj._indices()

        h =  torch.mm(x,self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes
        edge = torch.cat((h[indices[0]], h[indices[1]]), dim=1).t()

        edge = torch.exp(-self.leakyrelu(self.a.mm(edge).squeeze()))
        
        assert not torch.isnan(edge).any()
      
        rowsum = self.special_spmm(indices, edge, torch.Size([n, n]), torch.ones(size=(n,1)))
        
        edge =F.dropout(edge, self.dropout, training=self.training)

        h_prime = self.special_spmm(indices, edge, torch.Size([n, n]), h)
        
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(rowsum)

        assert not torch.isnan(h_prime).any()
        
        if self.res_connection:
            h_prime=h_prime+torch.mm(x,self.res)
            
        if self.apply_act:
  
            return F.elu(h_prime)

        else:

            return h_prime
    def init_weights(self,layers):
        # xavier weights initialization
        for layer in layers:
            nn.init.xavier_uniform(layer.weight)

