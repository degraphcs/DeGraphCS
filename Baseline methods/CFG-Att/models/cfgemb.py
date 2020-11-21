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
from modules import GGNN, SeqEncoder

class CFGEmbeder(nn.Module):
    def __init__(self, config):
        super(CFGEmbeder, self).__init__()
        self.conf = config

        self.margin = config['margin']
        self.n_desc_words = config['n_desc_words']
        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.dropout = config['dropout']
        self.cfg_attn_mode = config['cfg_attn_mode']

        self.cfg_encoder = GGNN(self.conf)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

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
     

    def code_encoding(self, cfg_init_input_batch, cfg_adjmat_batch, cfg_node_mask):
        
        batch_size = cfg_node_mask.size()[0]

        # code_feat: [batch_size x n_node x state_dim]
        code_feat = self.cfg_encoder(cfg_init_input_batch, cfg_adjmat_batch, cfg_node_mask) # forward(prop_state, A, node_mask)

        node_num = code_feat.size()[1] # n_node
        code_feat = code_feat.reshape(-1, node_num, self.n_hidden) # useless...
        # mask_1forgt0: [batch_size x n_node]
        mask_1forgt0 = cfg_node_mask.bool().reshape(-1, node_num) 

        if self.conf['transform_every_modal']:
            code_feat = torch.tanh(
                self.linear_single_modal(F.dropout(code_feat, self.dropout, training=self.training)))

        code_sa_tanh = F.tanh(self.self_atten(code_feat.reshape(-1, self.n_hidden))) # [(batch_size * n_node) x n_hidden]
        code_sa_tanh = F.dropout(code_sa_tanh, self.dropout, training=self.training)
        # code_sa_tanh: [batch_size x n_node]
        code_sa_tanh = self.self_atten_scalar(code_sa_tanh).reshape(-1, node_num) 
        
        code_feat = code_feat.reshape(-1, node_num, self.n_hidden)
        batch_size = code_feat.size()[0]

        self_attn_code_feat = None
        for _i in range(batch_size):
            # code_sa_tanh_one: [1 x real_node_num]
            code_sa_tanh_one = torch.masked_select(code_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)
            
            if self.cfg_attn_mode == 'sigmoid_scalar':
                # attn_w_one: [1 x 1 x real_node_num]
                attn_w_one = torch.sigmoid(code_sa_tanh_one).reshape(1, 1, -1)
            else:
                attn_w_one = F.softmax(code_sa_tanh_one, dim=1).reshape(1, 1, -1)
            
            if self.conf['save_attn_weight']:
                self.attn_weight_torch.append(attn_w_one.detach().reshape(1, -1).cpu())
                self.node_mask_torch.append(mask_1forgt0[_i, :].detach().reshape(1, -1).cpu())
            
            # attn_feat_one: [1 x real_node_num x n_hidden]
            attn_feat_one = torch.masked_select(code_feat[_i, :, :].reshape(1, node_num, self.n_hidden),
                                                mask_1forgt0[_i, :].reshape(1, node_num, 1)).reshape(1, -1, self.n_hidden)
            # out_to_cat: [1 x n_hidden]
            out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)   
            # self_attn_code_feat: [batch_size x n_hidden]                                                
            self_attn_code_feat = out_to_cat if self_attn_code_feat is None else torch.cat(
                (self_attn_code_feat, out_to_cat), 0)

        
        self_attn_code_feat = torch.tanh(self_attn_code_feat)
        '''
        self_attn_code_feat = torch.tanh(
            self.linear_attn_out(
                F.dropout(self_attn_code_feat, self.dropout, training=self.training))
        )
        '''
        # self_attn_code_feat: [batch_size x n_hidden]
        return self_attn_code_feat
        
    def desc_encoding(self, desc, desc_len):
        batch_size = desc.size(0)
        desc_enc_hidden = self.desc_encoder.init_hidden(batch_size)
        # desc_enc_hidden: [2 x batch_size x n_hidden]
        _, desc_enc_hidden = self.desc_encoder(desc, desc_len)
        # desc_feat: [batch_size x n_hidden]
        desc_feat = desc_enc_hidden[0].reshape(batch_size, self.n_hidden)

        if self.conf['transform_every_modal']:
            desc_feat = torch.tanh(
                self.linear_single_modal(
                    F.dropout(desc_feat, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']: 
            desc_feat = torch.tanh(desc_feat)

        # desc_feat: [batch_size x n_hidden]
        return desc_feat
    
    
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
    
    def forward(self, cfg_init_input_batch, cfg_adjmat_batch, cfg_node_mask, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_sz x n_hidden]  
        cfg_repr = self.code_encoding(cfg_init_input_batch, cfg_adjmat_batch, cfg_node_mask)
        # desc_repr: [batch_sz x n_hidden]  
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_sz]
        anchor_sim = self.similarity(cfg_repr, desc_anchor_repr)
        neg_sim = self.similarity(cfg_repr, desc_neg_repr) 
        
        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss
