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
        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.dropout = config['dropout']

        self.n_desc_words = config['n_desc_words']
        self.n_token_words = config['n_token_words']


        self.dfg_encoder = GGNN(self.conf)
        self.cfg_encoder = GGNN(self.conf)
        self.tok_encoder = SeqEncoder(self.n_token_words, self.emb_size, self.n_hidden)
        self.desc_encoder = SeqEncoder(self.n_desc_words, self.emb_size, self.n_hidden)

        self.tok_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.tok_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.dfg_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.dfg_attn_scalar = nn.Linear(self.n_hidden, 1)
        self.cfg_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.cfg_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.self_attn2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.self_attn_scalar2 = nn.Linear(self.n_hidden, 1)

        self.attn_modal_fusion = nn.Linear(self.n_hidden * 3, self.n_hidden)
     

    def code_encoding(self, tokens, tok_len, dfg_init_input, dfg_adjmat, dfg_node_mask, cfg_init_input, cfg_adjmat, cfg_node_mask):
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

        
        ''' DFG Embedding w.Attention '''
        # dfg_feat: [batch_size x n_node x state_dim]
        dfg_feat = self.dfg_encoder(dfg_init_input, dfg_adjmat, dfg_node_mask) # forward(prop_state, A, node_mask)

        node_num = dfg_feat.size()[1] # n_node
        dfg_feat = dfg_feat.reshape(-1, node_num, self.n_hidden)
        # dfg_mask_1forgt0: [batch_size x n_node]
        dfg_mask_1forgt0 = dfg_node_mask.bool().reshape(-1, node_num) 

        dfg_sa_tanh = F.tanh(self.dfg_attn(dfg_feat.reshape(-1, self.n_hidden))) # [(batch_size * n_node) x n_hidden]
        dfg_sa_tanh = F.dropout(dfg_sa_tanh, self.dropout, training=self.training)
        # dfg_sa_tanh: [batch_size x n_node]
        dfg_sa_tanh = self.dfg_attn_scalar(dfg_sa_tanh).reshape(-1, node_num) 
        dfg_feat = dfg_feat.reshape(-1, node_num, self.n_hidden)

        dfg_feat_attn = None
        for _i in range(batch_size):
            # dfg_sa_tanh_one: [1 x real_node_num]
            dfg_sa_tanh_one = torch.masked_select(dfg_sa_tanh[_i, :], dfg_mask_1forgt0[_i, :]).reshape(1, -1)        
            # attn_w_one: [1 x 1 x real_node_num]
            attn_w_one = torch.sigmoid(dfg_sa_tanh_one).reshape(1, 1, -1)      
            # attn_feat_one: [1 x real_node_num x n_hidden]
            attn_feat_one = torch.masked_select(dfg_feat[_i, :, :].reshape(1, node_num, self.n_hidden),
                                                dfg_mask_1forgt0[_i, :].reshape(1, node_num, 1)).reshape(1, -1, self.n_hidden)
            # out_to_cat: [1 x n_hidden]
            out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)   
            # dfg_feat_attn: [batch_size x n_hidden]                                                
            dfg_feat_attn = out_to_cat if dfg_feat_attn is None else torch.cat(
                (dfg_feat_attn, out_to_cat), 0)
        
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
        concat_feat = torch.cat((tok_feat_attn, dfg_feat_attn, cfg_feat_attn), 1)
        # code_feat: [batch_size x n_hidden]
        code_feat = torch.tanh(
            self.attn_modal_fusion(F.dropout(concat_feat, self.dropout, training=self.training))
        ).reshape(-1, self.n_hidden)

        return code_feat
        
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
                # self_attn_cfg_feat: [batch_sz x n_hidden]                                                
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
    
    def forward(self, tokens, tok_len, dfg_init_input, dfg_adjmat, dfg_node_mask, cfg_init_input, cfg_adjmat, cfg_node_mask, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):
        # code_repr: [batch_sz x n_hidden]  
        code_repr = self.code_encoding(tokens, tok_len, dfg_init_input, dfg_adjmat, dfg_node_mask, cfg_init_input, cfg_adjmat, cfg_node_mask)
        # desc_repr: [batch_sz x n_hidden]  
        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        # sim: [batch_sz]
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr) 
        
        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        
        return loss