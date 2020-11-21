import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
    
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node*self.n_edge_types]
        A_out = A[:, :, self.n_node*self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    """
    def __init__(self, config):
        super(GGNN, self).__init__()

        assert (config['state_dim'] >= config['annotation_dim'],  \
                'state_dim must be no less than annotation_dim')

        self.config = config

        self.vocab_size = config['n_ir_words']
        self.annotation_dim = config['annotation_dim']
        self.state_dim = config['state_dim']
        self.n_edge_types = config['n_edge_types']
        self.n_node = config['n_node']
        self.n_steps = config['n_steps']
        self.word_split = config['word_split']
        self.pooling_type = config['pooling_type']
        self.batch_size = config['batch_size']
        self.max_word_num = config['max_word_num']

        self.embedding = nn.Embedding(self.vocab_size, self.annotation_dim, padding_idx=0)
        
        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        # Output Model
        self.out_mlp = nn.Sequential(
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_xavier_linear(m)
    
    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def forward(self, annotation, A):
        
        # annotation: [batch_size x n_node x max_word_num_one_node] -> [batch_size x n_node x annotation_dim]    
        if self.word_split:
            if self.pooling_type == 'max_pooling':
                annotation = self.embedding(annotation)
                annotation = F.max_pool2d(annotation, kernel_size=(self.max_word_num,1), stride=1).squeeze(2)
            else: # 'ave_pooling'
                annotation = self.embedding(annotation)
                annotation = F.avg_pool2d(annotation, kernel_size=(self.max_word_num,1), stride=1).squeeze(2)

        # annotation: [batch_size x n_node] -> [batch_size x n_node x annotation_dim]
        else:
            annotation = self.embedding(annotation)

        # prop_state: [batch_size x n_node x state_dim]
        padding = torch.zeros(len(annotation), self.n_node, self.state_dim-self.annotation_dim).float().cuda()
        prop_state = torch.cat((annotation, padding), 2).cuda()
        # A: [batch_size x n_node x (n_node * n_edge_types * 2)]

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            # before in_states: [n_edge_types x batch_size x n_node x state_dim] -> [batch_size x n_edge_types x ...]
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            # after in_states: [batch_size x (n_node * n_edge_types) x state_dim]
            in_states = in_states.reshape(-1, self.n_node*self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.reshape(-1, self.n_node*self.n_edge_types, self.state_dim)

            # prop_state: [batch_size x n_node x state_dim]
            prop_state = self.propogator(in_states, out_states, prop_state, A)

        # when output_type is 'no_reduce'
        join_state = torch.cat((prop_state, annotation), 2)
        output = self.out_mlp(join_state)

        # output: [batch_size x n_node x state_dim]
        return output


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.init_xavier_linear(self.embedding, init_bias=False)

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=False)

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
