import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle

from utils import PAD_ID, UNK_ID, indexes2sent
import configs
from util_cfg import get_one_cfg_npy_info
from util_dfg import get_one_dfg_npy_info

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, config, data_dir, f_token, max_tok_len, f_dfgs, dfg_max_node_num, f_cfgs, cfg_max_node_num, f_descs=None, max_desc_len=None):
        self.max_tok_len = max_tok_len
        self.dfg_max_node_num = dfg_max_node_num
        self.cfg_max_node_num = cfg_max_node_num
        self.max_desc_len = max_desc_len
        self.n_edge_types = config['n_edge_types']
        self.state_dim = config['state_dim']
        #self.annotation_dim = config['annotation_dim']

        print("Loading Data...")

        self.dfg_mark_list = []
        start_index, end_index = [0, 0]
        with open(data_dir+f_dfgs, 'r') as dfg_file:
            self.dfg_lines = dfg_file.readlines()
            for i in range(0, len(self.dfg_lines)):
                self.dfg_lines[i] = self.dfg_lines[i].rstrip('\n')
                if self.dfg_lines[i][0:10] == 'BeginFunc:' and i != 0:
                    end_index = i
                    self.dfg_mark_list.append([start_index, end_index])
                    start_index = i
            self.dfg_mark_list.append([start_index, len(self.dfg_lines)])

        self.cfg_mark_list = []
        start_index, end_index = [0, 0]
        with open(data_dir+f_cfgs, 'r') as cfg_file:
            self.cfg_lines = cfg_file.readlines()
            for i in range(0, len(self.cfg_lines)):
                self.cfg_lines[i] = self.cfg_lines[i].rstrip('\n')
                if self.cfg_lines[i][0:10] == 'BeginFunc:' and i != 0:
                    end_index = i
                    self.cfg_mark_list.append([start_index, end_index])
                    start_index = i
            self.cfg_mark_list.append([start_index, len(self.cfg_lines)])
        
        table_tokens = tables.open_file(data_dir+f_token)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]

        table_desc = tables.open_file(data_dir+f_descs)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]
        
        assert self.idx_tokens.shape[0]==self.idx_descs.shape[0]
        assert self.idx_tokens.shape[0]==len(self.dfg_mark_list)
        assert self.idx_tokens.shape[0]==len(self.cfg_mark_list)

        self.data_len = self.idx_descs.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):          
        input_dfg_lines = self.dfg_lines[self.dfg_mark_list[offset][0]: self.dfg_mark_list[offset][1]]
        dfg_init_input, dfg_adjmat, dfg_node_mask = get_one_dfg_npy_info(input_dfg_lines, 
                    self.dfg_max_node_num, self.n_edge_types, self.state_dim)

        input_cfg_lines = self.cfg_lines[self.cfg_mark_list[offset][0]: self.cfg_mark_list[offset][1]]
        cfg_init_input, cfg_adjmat, cfg_node_mask = get_one_cfg_npy_info(input_cfg_lines, 
                    self.cfg_max_node_num, self.n_edge_types, self.state_dim)
            
        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos+tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)

        len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
        good_desc_len = min(int(len), self.max_desc_len)
        good_desc = self.descs[pos: pos+good_desc_len]
        good_desc = self.pad_seq(good_desc, self.max_desc_len)
        
        rand_offset = random.randint(0, self.data_len-1)
        len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
        bad_desc_len = min(int(len), self.max_desc_len)
        bad_desc = self.descs[pos: pos+bad_desc_len]
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        return tokens, tok_len, torch.Tensor(dfg_init_input), torch.Tensor(dfg_adjmat), torch.Tensor(dfg_node_mask), torch.Tensor(cfg_init_input), torch.Tensor(cfg_adjmat), torch.Tensor(cfg_node_mask), good_desc, good_desc_len, bad_desc, bad_desc_len
        
    def __len__(self):
        return self.data_len

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 


if __name__ == '__main__':
    device = 'cpu'
    config = getattr(configs, 'config_CFGEmbeder')()
    input_dir = './data/github/'

    train_set = CodeSearchDataset(config, input_dir, 'train.token.h5', 100, 'train.dfg.txt', 512, 'train.cfg.txt', 512, 'train.desc.h5', 30)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=5, shuffle=False, drop_last=False, num_workers=1)
    print('number of batch:\n', len(train_data_loader))
    '''
    use_set = CodeSearchDataset(input_dir, 'use.tokens.h5', 30)
    use_data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1, shuffle=False, num_workers=1)
    #print(len(use_data_loader))
    vocab_tokens = load_dict(input_dir+'vocab.tokens.json')
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    '''
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    print('============ Train Data ================')
    k = 0
    for epo in range(0,3):
        for batch in train_data_loader:
            print("batch[1].size(): ", batch[1].size())
            #batch = tuple([t.numpy() for t in batch])
            cfg_init_input, cfg_adjmat, cfg_node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len = [tensor.to(device) for tensor in batch]
            print(cfg_adjmat.dtype)
            #print(batch)
            k+=1
            #if k>0: break
            print('-------------------------------')
            print(indexes2sent(good_desc, vocab_desc))
            #print(indexes2sent(good_desc, vocab_desc))
    
    
