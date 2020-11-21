import os
import numpy as np
import math
import dgl

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].reshape(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).reshape(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return_iou = self.U_iou(h_cat)

        return {'iou': return_iou, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self, config):
        super(TreeLSTM, self).__init__()
        
        self.vocab_size = config['n_ast_words']
        self.emb_size = config['emb_size']
        self.hidden_size = config['n_hidden']
        self.treelstm_cell_type = config['treelstm_cell_type']

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)
        
        self.init_xavier_linear(self.embedding, init_bias=False)
        
        cell = TreeLSTMCell if self.treelstm_cell_type == 'nary' else ChildSumTreeLSTMCell
        self.cell = cell(self.emb_size, self.hidden_size)

    def forward(self, batch, enc_hidden):

        g = batch.graph
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)

        # feed embedding
        g.ndata['h'] = enc_hidden[0] # h: initial hidden state 
        g.ndata['c'] = enc_hidden[1] # c: initial cell state

        #print(batch.wordid.shape)
        #print(batch.mask.shape)
        #print(batch.words.shape)
        # batch_wordid: [tree_node x 5]->[tree_node x 5 x anno_dim]
        
        embeds = self.embedding(batch.words)
        #print('embeds shape: ', embeds.shape)
        mask = batch.mask.unsqueeze(1).unsqueeze(2)
        #print('mask shape: ', mask.shape)
        embeds = embeds * mask
        embeds = F.avg_pool2d(embeds, kernel_size=(5,1), stride=1).squeeze(1)
        #print('after embeds shape:', embeds.shape)
        
        #embeds = self.embedding(batch.wordid * batch.mask)

        g.ndata['iou'] = self.cell.W_iou(embeds) * batch.mask.float().unsqueeze(-1)

        # propagate
        dgl.prop_nodes_topo(g)

        all_node_h_in_batch = g.ndata.pop('h') # no dropout so far
        all_node_c_in_batch = g.ndata.pop('c')

        # when treelstm_output_type is "no_reduce"

        # all_node_h_in_batch: [all_node_num_in_batch x hidden_size]
        return all_node_h_in_batch, all_node_c_in_batch

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (weight.new(batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(batch_size, self.hidden_size).zero_().requires_grad_())

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

        
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.init_xavier_linear(self.embedding, init_bias=False)

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0.1, batch_first=True, bidirectional=False)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_(), # rnn_type == 'LSTM'
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_())


    def forward(self, inputs, input_lens=None, hidden=None): 
        batch_size, seq_len = inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        #inputs = F.dropout(inputs, 0.1, self.training) # mark.
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
            
        hids, (h_n, c_n) = self.lstm(inputs, hidden) # hids:[b x seq x hid_sz*2](biRNN) 
        
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)   
            #hids = F.dropout(hids, p=0.1, training=self.training) # mark.
            hids = hids.index_select(0, inv_indices) # [batch_sz x seq_len x hid_sz]
            h_n = h_n.index_select(1, inv_indices)
            c_n = c_n.index_select(1, inv_indices)

        h_n = h_n[0] # [batch_sz x hid_sz] n_layers==1 and n_dirs==1
        c_n = c_n[0] 

        return hids, (h_n, c_n)

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)
