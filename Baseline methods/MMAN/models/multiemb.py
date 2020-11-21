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
sys.path.insert(0, parentPath)
from modules import SeqEncoder, GGNN, TreeLSTM

class MultiEmbeder(nn.Module):
    def __init__(self, config):
        super(MultiEmbeder, self).__init__()
        self.conf = config

        self.margin = config['margin']
        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.dropout = config['dropout']

        self.n_desc_words = config['n_desc_words']
        self.n_token_words = config['n_token_words']

        self.ast_encoder = TreeLSTM(self.conf)
        self.cfg_encoder = GGNN(self.conf)
        self.tok_encoder = SeqEncoder(self.n_token_words, self.emb_size, self.n_hidden)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

        self.tok_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.tok_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.ast_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.ast_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.cfg_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.cfg_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.attn_modal_fusion = nn.Linear(self.n_hidden * 3, self.n_hidden)

    def code_encoding(self, tokens, tok_len, tree, tree_node_num, cfg_init_input, cfg_adjmat, cfg_node_mask):
        
        batch_size = cfg_node_mask.size()[0]

        ''' Token Embedding w.Attention '''
        tok_enc_hidden = self.tok_encoder.init_hidden(batch_size)
        # tok_feat: [batch_size x seq_len x hidden_size]
        tok_feat, _ = self.tok_encoder(tokens, tok_len, tok_enc_hidden)

        seq_len = tok_feat.size()[1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        tok_unpack_len_list = tok_len.long().to(device)
        range_tensor = torch.arange(seq_len).to(device)
        tok_mask_1forgt0 = range_tensor[None, :] < tok_unpack_len_list[:, None] 
        tok_mask_1forgt0 = tok_mask_1forgt0.reshape(-1, seq_len)
        
        tok_sa_tanh = torch.tanh(self.tok_attn(tok_feat.reshape(-1, self.n_hidden))) 
        tok_sa_tanh = F.dropout(tok_sa_tanh, self.dropout, training=self.training)
        # tok_sa_tanh: [batch_size x seq_len]
        tok_sa_tanh = self.tok_attn_scalar(tok_sa_tanh).reshape(-1, seq_len) 
        tok_feat = tok_feat.reshape(-1, seq_len, self.n_hidden)
        
        tok_feat_attn = None
        for _i in range(batch_size):
            tok_sa_tanh_one = torch.masked_select(tok_sa_tanh[_i, :], tok_mask_1forgt0[_i, :]).reshape(1, -1)
            # attn_w_one: [1 x 1 x seq_real_len]
            attn_w_one = F.softmax(tok_sa_tanh_one, dim=1).reshape(1, 1, -1)
            # attn_feat_one: [1 x seq_real_len x n_hidden]
            attn_feat_one = torch.masked_select(tok_feat[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                tok_mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1, self.n_hidden)
            # out_to_cat: [1 x n_hidden]
            out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)   
            # tok_feat_attn: [batch_sz x n_hidden]                                                
            tok_feat_attn = out_to_cat if tok_feat_attn is None else torch.cat(
                (tok_feat_attn, out_to_cat), 0)     
        
        ''' AST Embedding w.Attention '''
        # tree: contain ['graph', 'mask', 'wordid', 'label']
        ast_enc_hidden = self.ast_encoder.init_hidden(tree.graph.number_of_nodes()) # use all_node_num to initialize h_0, c_0
        # all_node_h/c_in_batch: [all_node_num_in_batch x hidden_size]
        all_node_h_in_batch, all_node_c_in_batch = self.ast_encoder(tree, ast_enc_hidden)
        
        ast_feat_attn = None
        add_up_node_num = 0
        for _i in range(batch_size):
            # this_sample_h: [this_sample_node_num x hidden_size]
            this_sample_h = all_node_h_in_batch[add_up_node_num: add_up_node_num + tree_node_num[_i]]
            add_up_node_num += tree_node_num[_i]

            node_num = tree_node_num[_i] # this_sample_node_num
            ast_sa_tanh = torch.tanh(self.ast_attn(this_sample_h.reshape(-1, self.n_hidden)))
            ast_sa_tanh = F.dropout(ast_sa_tanh, self.dropout, training=self.training)
            ast_sa_before_softmax = self.ast_attn_scalar(ast_sa_tanh).reshape(1, node_num)
            # ast_attn_weight: [1 x this_sample_node_num]
            ast_attn_weight = F.softmax(ast_sa_before_softmax, dim=1)      
            # ast_attn_this_sample_h: [1 x n_hidden]
            ast_attn_this_sample_h = torch.bmm(ast_attn_weight.reshape(1, 1, node_num),
                                                this_sample_h.reshape(1, node_num, self.n_hidden)).reshape(1, self.n_hidden)
            # ast_feat_attn: [batch_size x n_hidden]
            ast_feat_attn = ast_attn_this_sample_h if ast_feat_attn is None else torch.cat(
                (ast_feat_attn, ast_attn_this_sample_h), 0)

        ''' CFG Embedding w.Attention '''
        # cfg_feat: [batch_size x n_node x state_dim]
        cfg_feat = self.cfg_encoder(cfg_init_input, cfg_adjmat, cfg_node_mask) # forward(prop_state, A, node_mask)

        node_num = cfg_feat.size()[1] # n_node
        cfg_feat = cfg_feat.reshape(-1, node_num, self.n_hidden)
        # cfg_mask_1forgt0: [batch_size x n_node]
        cfg_mask_1forgt0 = cfg_node_mask.bool().reshape(-1, node_num) 

        cfg_sa_tanh = F.tanh(self.cfg_attn(cfg_feat.reshape(-1, self.n_hidden))) # [(batch_size * n_node) x n_hidden]
        cfg_sa_tanh = F.dropout(cfg_sa_tanh, self.dropout, training=self.training)
        # cfg_sa_tanh: [batch_size x n_node]
        cfg_sa_tanh = self.cfg_attn_scalar(cfg_sa_tanh).reshape(-1, node_num) 
        cfg_feat = cfg_feat.reshape(-1, node_num, self.n_hidden)

        cfg_feat_attn = None
        for _i in range(batch_size):
            # cfg_sa_tanh_one: [1 x real_node_num]
            cfg_sa_tanh_one = torch.masked_select(cfg_sa_tanh[_i, :], cfg_mask_1forgt0[_i, :]).reshape(1, -1)        
            # attn_w_one: [1 x 1 x real_node_num]
            attn_w_one = torch.sigmoid(cfg_sa_tanh_one).reshape(1, 1, -1)      
            # attn_feat_one: [1 x real_node_num x n_hidden]
            attn_feat_one = torch.masked_select(cfg_feat[_i, :, :].reshape(1, node_num, self.n_hidden),
                                                cfg_mask_1forgt0[_i, :].reshape(1, node_num, 1)).reshape(1, -1, self.n_hidden)
            # out_to_cat: [1 x n_hidden]
            out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)   
            # cfg_feat_attn: [batch_size x n_hidden]                                                
            cfg_feat_attn = out_to_cat if cfg_feat_attn is None else torch.cat(
                (cfg_feat_attn, out_to_cat), 0)

        # concat_feat: [batch_size x (n_hidden * 3)]
        concat_feat = torch.cat((tok_feat_attn, ast_feat_attn, cfg_feat_attn), 1)
        # code_feat: [batch_size x n_hidden]
        code_feat = torch.tanh(
            self.attn_modal_fusion(F.dropout(concat_feat, self.dropout, training=self.training))
        ).reshape(-1, self.n_hidden)

        return code_feat
        
    def desc_encoding(self, desc, desc_len):
        batch_size = desc.size(0)

        desc_enc_hidden = self.desc_encoder.init_hidden(batch_size)
        # desc_enc_hidden: [2 x batch_size x n_hidden]
        _, desc_enc_hidden = self.desc_encoder(desc, desc_len)
        # desc_feat: [batch_size x n_hidden]
        desc_feat = desc_enc_hidden[0].reshape(batch_size, self.n_hidden)
        # desc_feat: [batch_size x n_hidden]
        desc_feat = torch.tanh(desc_feat)
        
        return desc_feat
    
    
    def forward(self, tokens, tok_len, tree, tree_node_num, cfg_init_input, cfg_adjmat, cfg_node_mask, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_size x n_hidden]  
        code_repr = self.code_encoding(tokens, tok_len, tree, tree_node_num, cfg_init_input, cfg_adjmat, cfg_node_mask)
        # desc_repr: [batch_size x n_hidden]  
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_size]
        anchor_sim = F.cosine_similarity(code_repr, desc_anchor_repr)
        neg_sim = F.cosine_similarity(code_repr, desc_neg_repr) 
        
        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss