import sys
import torch 
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
import collections
import dgl

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import configs
from data_prepare.util_cfg import get_cfg_npy_info, get_one_cfg_npy_info
from data_prepare.util_ast import build_tree

PAD_ID, UNK_ID = [0, 1]

ASTBatch = collections.namedtuple('ASTBatch', ['graph', 'mask', 'wordid', 'label'])
def batcher(device):
    def batcher_dev(batch):
        # all variable should be added '_batch'
        tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len = zip(*batch)
        
        tree = dgl.batch(tree)
        tree = ASTBatch(graph=tree,
                        mask=tree.ndata['mask'].to(device),
                        wordid=tree.ndata['x'].to(device),
                        label=tree.ndata['y'].to(device))

        tokens = tuplelist2tensor_long(tokens).to(device)
        tok_len = tuple2tensor_long(tok_len).to(device)

        tree_node_num = tuple2tensor_long(tree_node_num).to(device)
        init_input = tuple3list2tensor_float(init_input).to(device)
        adjmat = tuple3list2tensor_float(adjmat).to(device)
        node_mask = tuplelist2tensor_long(node_mask).to(device)

        good_desc = tuplelist2tensor_long(good_desc).to(device)
        good_desc_len = tuple2tensor_long(good_desc_len).to(device)
        bad_desc = tuplelist2tensor_long(bad_desc).to(device)
        bad_desc_len = tuple2tensor_long(bad_desc_len).to(device)
        
        return tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len
    return batcher_dev

class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, config, data_dir, f_token, max_tok_len, f_ast, f_ast_dict, f_cfg, max_node_num, f_desc, max_desc_len):
        self.max_tok_len = max_tok_len
        self.max_node_num = max_node_num
        self.max_desc_len = max_desc_len

        self.n_edge_types = config['n_edge_types']
        self.state_dim = config['state_dim']
        self.annotation_dim = config['annotation_dim']

        self.trees = []
        self.trees_num = []
        
        print("Loading Data...")

        # loading AST data
        ast_tree_json = json.loads(open(data_dir+f_ast, 'r').readline())
        vacab_ast_dict = json.loads(open(data_dir+f_ast_dict, 'r').readline())

        for i in range(0, len(ast_tree_json)):
            tree_json = ast_tree_json[str(i)]
            self.trees.append(build_tree(tree_json, vacab_ast_dict))
            self.trees_num.append(self.trees[i].number_of_nodes())

        # loading CFG data
        self.mark_list = []
        start_index, end_index = [0, 0]
        with open(data_dir+f_cfg, 'r') as cfg_file:
            self.cfg_lines = cfg_file.readlines()
            for i in range(0, len(self.cfg_lines)):
                self.cfg_lines[i] = self.cfg_lines[i].rstrip('\n')
                if self.cfg_lines[i][0:10] == 'BeginFunc:' and i != 0:
                    end_index = i
                    self.mark_list.append([start_index, end_index])
                    start_index = i
            self.mark_list.append([start_index, len(self.cfg_lines)])

        # loading Token data
        table_tokens = tables.open_file(data_dir+f_token)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]

        # loading Description data
        table_desc = tables.open_file(data_dir+f_desc)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]
        
        assert self.idx_tokens.shape[0]==self.idx_descs.shape[0]
        assert self.idx_tokens.shape[0]==len(self.trees)
        assert self.idx_tokens.shape[0]==len(self.mark_list)

        self.data_len = self.idx_descs.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):      
        
        # tree:contain ['graph', 'mask', 'wordid', 'label'], tree_node_num:n
        tree = self.trees[offset]
        tree_node_num = self.trees_num[offset]
        
        # init_input:[n_node x state_dim], adjmat:[n_node x (n_node*n_edge_types*2)], node_mask:[n_node]
        input_cfg_lines = self.cfg_lines[self.mark_list[offset][0]: self.mark_list[offset][1]]
        init_input, adjmat, node_mask = get_one_cfg_npy_info(input_cfg_lines, 
                    self.max_node_num, self.n_edge_types, self.state_dim, self.annotation_dim)

        # tokens:[max_tok_len], tok_len:n
        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos+tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)

        # desc:[max_desc_len], desc_len:n
        len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
        good_desc_len = min(int(len), self.max_desc_len)
        good_desc = self.descs[pos: pos+good_desc_len]
        good_desc = self.pad_seq(good_desc, self.max_desc_len)
        
        rand_offset = random.randint(0, self.data_len-1)
        len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
        bad_desc_len = min(int(len), self.max_desc_len)
        bad_desc = self.descs[pos: pos+bad_desc_len]
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        return tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len
        
    def __len__(self):
        return self.data_len


def tuple2tensor_long(long_tuple):
    long_numpy = np.zeros(len(long_tuple))
    for index, value in enumerate(long_tuple):
        long_numpy[index] = value
    long_tensor = torch.from_numpy(long_numpy).type(torch.LongTensor)
    return long_tensor

def tuplelist2tensor_long(long_tuple_list):
    long_numpy = np.zeros([len(long_tuple_list), len(long_tuple_list[0])])
    for index, value in enumerate(long_tuple_list):
        long_numpy[index] = value
    long_tensor = torch.from_numpy(long_numpy).type(torch.LongTensor)
    return long_tensor

def tuple3list2tensor_float(float_tuple_3list):
    float_numpy = np.zeros([len(float_tuple_3list), float_tuple_3list[0].shape[0], float_tuple_3list[0].shape[1]])
    for index, value in enumerate(float_tuple_3list):
        float_numpy[index] = value
    float_tensor = torch.from_numpy(float_numpy).float()
    return float_tensor


def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 


if __name__ == '__main__':
    device = 'cpu'
    config = getattr(configs, 'config_MultiEmbeder')()
    input_dir = './data/github/'
    
    train_set = CodeSearchDataset(config, input_dir, 
                        'train.token.h5', 50, 'train.ast.json', 'vocab.ast.json', 'train.cfg.txt', 512, 'train.desc.h5', 30)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=5, 
                        collate_fn=batcher(device), shuffle=False, drop_last=False, num_workers=1)
    print('number of batch:\n', len(train_data_loader))
    '''
    use_set = CodeSearchDataset(input_dir, 'use.tokens.h5', 30)
    use_data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1, shuffle=False, num_workers=1)
    #print(len(use_data_loader))
    vocab_tokens = load_dict(input_dir+'vocab.tokens.json')
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    '''
    #vocab_desc = load_dict(input_dir+'vocab.desc.json')
    print('============ Train Data ================')
    k = 0
    for batch in train_data_loader:
        print(batch)
        tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len = [t for t in batch]
        print(batch)
        k += 1
        if k>0: break
        print('-------------------------------')
        #print(indexes2sent(good_desc, vocab_desc))
        #print(indexes2sent(good_desc, vocab_desc))
    
    
