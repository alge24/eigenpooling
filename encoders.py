## The code is partially adapted from https://github.com/RexYing/diffpool

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True, device='cpu'):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout

        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout).to(device)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim)).to(device)
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x

        y = torch.matmul(y,self.weight)

        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y



class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, args=None, device='cpu'):
        super(GcnEncoderGraph, self).__init__()

        print('Whether concat', concat)

        self.device = device

        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU().to(device)
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

        print('num_layers: ', num_layers)
        print('pred_hidden_dims: ', pred_hidden_dims)
        print('hidden_dim: ', hidden_dim)
        print('embedding_dim: ', embedding_dim)
        print('label_dim', label_dim)



    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, device=self.device)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias, device=self.device) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias, device=self.device)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):


        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim).to(self.device)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim).to(self.device))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim).to(self.device))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes): 
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)).to(self.device) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes).to(self.device)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2)

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)

        if self.bn:
            x = self.apply_bn(x)

        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)

        x = conv_last(x,adj)
        x_all.append(x)

        if self.concat:
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x


        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):

        max_num_nodes = adj.size()[1]

        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None


        x = self.conv_first(x, adj)

        x = self.act(x)
 
        if self.bn:
            x = self.apply_bn(x)


    
        out_all = []
        out, _ = torch.max(x, dim=1)

        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x,adj)

        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        ypred = self.pred_model(output)

        return ypred

    def loss(self, pred, label, type='softmax'):

        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().to(self.device)
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot).to(self.device)
            










class WavePoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
             num_pool_matrix=2, num_pool_final_matrix=0, pool_sizes = [4] , pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0 , mask=1,args=None, device='cpu'):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(WavePoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args, device=device)
        add_self = not concat
        self.mask = mask
        self.pool_sizes = pool_sizes
        self.num_pool_matrix = num_pool_matrix
        self.num_pool_final_matrix = num_pool_final_matrix

        self.con_final = args.con_final

        self.device = device

        print('Device_-wave: ', device )

        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(len(pool_sizes)):


            print('In WavePooling',self.pred_input_dim*self.num_pool_matrix)
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim*self.num_pool_matrix, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)


            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)
            


        if self.num_pool_final_matrix > 0:
            if concat:

                if self.con_final:
                    self.pred_model = self.build_pred_layers(self.pred_input_dim * (len(pool_sizes)+1) + self.pred_input_dim*self.num_pool_final_matrix, pred_hidden_dims, 
                            label_dim, num_aggs=self.num_aggs)
                else:
                    self.pred_model = self.build_pred_layers(self.pred_input_dim * (len(pool_sizes)) + self.pred_input_dim*self.num_pool_final_matrix, pred_hidden_dims, 
                            label_dim, num_aggs=self.num_aggs)
        

            else:


                self.pred_model = self.build_pred_layers( self.pred_input_dim*self.num_pool_final_matrix, pred_hidden_dims, 
                        label_dim, num_aggs=self.num_aggs)
               
        else:
            if concat:
                self.pred_model = self.build_pred_layers(self.pred_input_dim * (len(pool_sizes)+1), pred_hidden_dims, 
                        label_dim, num_aggs=self.num_aggs)
            else:
                self.pred_model = self.build_pred_layers(self.pred_input_dim , pred_hidden_dims, 
                        label_dim, num_aggs=self.num_aggs)
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)



    def forward(self, x, adj, adj_pooled_list, batch_num_nodes, batch_num_nodes_list, pool_matrices_dic ,  **kwargs):
 
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []


        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)


        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)



        for i in range(len(self.pool_sizes)):
            pool = Pool(self.num_pool_matrix, pool_matrices_dic[i], device=self.device)

            embedding_tensor = pool(embedding_tensor)
            if self.mask:
                embedding_mask =self.construct_mask(max_num_nodes, batch_num_nodes_list[i])
            else:
                embedding_mask = None
            adj_new = adj_pooled_list[i].type(torch.FloatTensor).to(self.device)
            embedding_tensor = self.gcn_forward(embedding_tensor, adj_new, 
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i], embedding_mask)

            if self.con_final or self.num_pool_final_matrix == 0:
                out, _ = torch.max(embedding_tensor, dim=1)
                out_all.append(out)


        if self.num_pool_final_matrix >0:

            pool = Pool(self.num_pool_final_matrix, pool_matrices_dic[i+1], device=self.device)
            embedding_tensor =  pool(embedding_tensor)
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)



        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        y_pred = self.pred_model(output)
        return y_pred

    def loss(self, pred, label):
        return F.cross_entropy(pred, label, size_average=True)



class Pool(nn.Module):
    def __init__(self, num_pool, pool_matrices, device = 'cpu'):
        super(Pool,self).__init__()

        self.pool_matrices = pool_matrices
        self.num_pool = num_pool


        self.device = device


    def forward(self,x):
        pooling_results = [0]*self.num_pool
        for i in range(self.num_pool):

            pool_matrix = self.pool_matrices[i]
            pool_matrix = pool_matrix.type(torch.FloatTensor).to(self.device)

 

            pool_matrix =  torch.transpose(pool_matrix, 1, 2)

            pooling_results[i] = torch.matmul(pool_matrix, x)
        if len(pooling_results)>1:

            x_pooled = torch.cat([*pooling_results],2)

        else:
            x_pooled = pooling_results[0]



        return x_pooled
