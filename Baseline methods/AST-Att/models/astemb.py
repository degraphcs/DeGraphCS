import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import SeqEncoder, TreeLSTM

class ASTEmbeder(nn.Module):
    def __init__(self, config):
        super(ASTEmbeder, self).__init__()
        self.conf = config
        
        self.margin = config['margin']
        self.n_desc_words = config['n_desc_words']
        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.dropout = config['dropout']

        self.ast_encoder = TreeLSTM(self.conf)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

        if self.conf['transform_attn_out']:
            self.linear_attn_out = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                nn.Tanh(),
                                                nn.Linear(self.n_hidden, self.n_hidden))

        if self.conf['transform_every_modal']:
            self.linear_single_modal = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                     nn.Tanh(),
                                                     nn.Linear(self.n_hidden, self.n_hidden))

        if self.conf['save_attn_weight']:
            self.attn_weight_torch = []
            self.node_mask_torch = []

        self.self_atten = nn.Linear(self.n_hidden, self.n_hidden)
        self.self_atten_scalar = nn.Linear(self.n_hidden, 1)
        self.self_attn2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.self_attn_scalar2 = nn.Linear(self.n_hidden, 1)
        
        #self.init_weights()
  

    def code_encoding(self, tree_batch, tree_node_num):
        batch_size = tree_node_num.size(0) # tree_num
        # tree_batch: contain ['graph', 'mask', 'wordid', 'label']
        code_enc_hidden = self.ast_encoder.init_hidden(tree_batch.graph.number_of_nodes()) # use all_node_num to initialize h_0, c_0
        # all_node_h_in_batch: [all_node_num_in_batch x hidden_size]
        all_node_h_in_batch, all_node_c_in_batch = self.ast_encoder(tree_batch, code_enc_hidden)

        if self.conf['transform_every_modal']:
            all_node_h_in_batch = torch.tanh(
                self.linear_single_modal(F.dropout(all_node_h_in_batch, self.dropout, training=self.training)))
                
        elif self.conf['use_tanh']:
            all_node_h_in_batch = torch.tanh(all_node_h_in_batch)
        
        self_attn_code_feat = None
        add_up_node_num = 0
        for _i in range(batch_size):
            # this_sample_h: [this_sample_node_num x hidden_size]
            this_sample_h = all_node_h_in_batch[add_up_node_num: add_up_node_num + tree_node_num[_i]]
            add_up_node_num += tree_node_num[_i]

            node_num = tree_node_num[_i] # this_sample_node_num
            code_sa_tanh = torch.tanh(self.self_atten(this_sample_h.reshape(-1, self.n_hidden)))
            code_sa_tanh = F.dropout(code_sa_tanh, self.dropout, training=self.training)

            code_sa_before_softmax = self.self_atten_scalar(code_sa_tanh).reshape(1, node_num)
            self_atten_weight = F.softmax(code_sa_before_softmax, dim=1)
            
            if self.conf['save_attn_weight']:
                self.attn_weight_torch.append(self_atten_weight.detach().reshape(1, -1).cpu())
            
            # self_attn_this_sample_h: [1 x hidden_size]
            self_attn_this_sample_h = torch.bmm(self_atten_weight.reshape(1, 1, node_num),
                                                this_sample_h.reshape(1, node_num, self.n_hidden)).reshape(1, self.n_hidden)
            
            if self_attn_code_feat is None:
                self_attn_code_feat = self_attn_this_sample_h
            else:
                self_attn_code_feat = torch.cat((self_attn_code_feat, self_attn_this_sample_h), 0)

        if self.conf['transform_attn_out']:
            # self_attn_code_feat: [batch_size x hidden_size]
            self_attn_code_feat = torch.tanh(
                self.linear_attn_out(
                    F.dropout(self_attn_code_feat, self.dropout, training=self.training)))
        
        elif self.conf['use_tanh']:
            self_attn_code_feat = torch.tanh(self_attn_code_feat)

        return self_attn_code_feat
        
 
    def desc_encoding(self, desc, desc_len):
        batch_size = desc.size()[0]
        desc_enc_hidden = self.desc_encoder.init_hidden(batch_size)
        # desc_enc_hidden: [2 x batch_size x n_hidden]
        desc_feat, desc_enc_hidden = self.desc_encoder(desc, desc_len, desc_enc_hidden)
        desc_enc_hidden = desc_enc_hidden[0]

        if self.conf['use_desc_attn']:
            seq_len = desc_feat.size()[1]

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            unpack_len_list = desc_len.long().to(device)
            range_tensor = torch.arange(seq_len).to(device)
            mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
            mask_1forgt0 = mask_1forgt0.reshape(-1, seq_len)

            desc_sa_tanh = torch.tanh(self.self_attn2(desc_feat.reshape(-1, self.n_hidden))) # [(batch_sz * seq_len) x n_hidden]
            desc_sa_tanh = F.dropout(desc_sa_tanh, self.dropout, training=self.training)
            desc_sa_tanh = self.self_attn_scalar2(desc_sa_tanh).reshape(-1, seq_len) # [batch_sz x seq_len]
            desc_feat = desc_feat.reshape(-1, seq_len, self.n_hidden)
            
            self_attn_desc_feat = None
            for _i in range(batch_size):
                desc_sa_tanh_one = torch.masked_select(desc_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)
                # attn_w_one: [1 x 1 x seq_len]
                attn_w_one = F.softmax(desc_sa_tanh_one, dim=1).reshape(1, 1, -1)
                
                # attn_feat_one: [1 x seq_len x n_hidden]
                attn_feat_one = torch.masked_select(desc_feat[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1, self.n_hidden)
                # out_to_cat: [1 x n_hidden]
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)   
                # self_attn_code_feat: [batch_sz x n_hidden]                                                
                self_attn_desc_feat = out_to_cat if self_attn_desc_feat is None else torch.cat(
                    (self_attn_desc_feat, out_to_cat), 0)

        else:
            self_attn_desc_feat = desc_enc_hidden.reshape(batch_size, self.n_hidden)
            
        if self.conf['use_tanh']:
            self_attn_desc_feat = torch.tanh(self_attn_desc_feat)
        
        # desc_feat: [batch_size x n_hidden]
        return self_attn_desc_feat
    
    
    def similarity(self, code_vec, desc_vec):
        assert self.conf['sim_measure'] in ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"
        if self.conf['sim_measure']=='cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.conf['sim_measure']=='poly':
            return (0.5*torch.matmul(code_vec, desc_vec.t()).diag()+1)**2
        elif self.conf['sim_measure']=='sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag()+1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2) # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure']=='euc': return euc_sim                
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag()+1)
            if self.conf['sim_measure']=='gesd': 
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure']=='aesd':
                return 0.5*(euc_sim+sigmoid_sim)
    
    def forward(self, tree_batch, tree_node_num, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_sz x n_hidden]  
        code_repr = self.code_encoding(tree_batch, tree_node_num)
        # desc_repr: [batch_sz x n_hidden]  
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_sz]
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr) 
        
        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss