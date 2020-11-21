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
from util_ast import build_tree
import collections
import dgl

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


ASTBatch = collections.namedtuple('ASTBatch', ['graph', 'mask', 'wordid', 'label', 'words'])
def batcher(device):
    def batcher_dev(batch):
        tree_batch, tree_node_num_batch, good_desc_batch, good_desc_len_batch, bad_desc_batch, bad_desc_len_batch = zip(*batch)
        
        tree_batch = dgl.batch(tree_batch)
        #print(tree_batch.ndata['x'])
        #print(tree_batch.ndata['words'])
        tree_batch = ASTBatch(graph=tree_batch,
                        mask=tree_batch.ndata['mask'].to(device),
                        wordid=tree_batch.ndata['x'].to(device),
                        label=tree_batch.ndata['y'].to(device),
                        words=tree_batch.ndata['words'].to(device))
        tree_node_num_batch = tuple2tensor_long(tree_node_num_batch).to(device)
        good_desc_batch = tuplelist2tensor_long(good_desc_batch).to(device)
        good_desc_len_batch = tuple2tensor_long(good_desc_len_batch).to(device)
        bad_desc_batch = tuplelist2tensor_long(bad_desc_batch).to(device)
        bad_desc_len_batch = tuple2tensor_long(bad_desc_len_batch).to(device)
        
        return tree_batch, tree_node_num_batch, good_desc_batch, good_desc_len_batch, bad_desc_batch, bad_desc_len_batch
    return batcher_dev

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_ast, f_ast_dict, f_descs=None, max_desc_len=None):
        
        self.max_desc_len = max_desc_len
        self.trees = []
        self.trees_num = []
        
        self.training = False
        print("loading data...")

        ast_file_path = data_dir + f_ast
        ast_dict_path = data_dir + f_ast_dict
        ast_tree_json = json.loads(open(ast_file_path, 'r').readline())
        vacab_ast_dict = json.loads(open(ast_dict_path, "r").readline())

        for i in range(0, len(ast_tree_json)):
            tree_json = ast_tree_json[str(i)]
            self.trees.append(build_tree(tree_json, vacab_ast_dict))
            self.trees_num.append(self.trees[i].number_of_nodes())
        
        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
        
        print("tree_num:\n", len(self.trees))
        print("desc_num:\n", self.idx_descs.shape[0])
        if f_descs is not None:
            assert len(self.trees)==self.idx_descs.shape[0]
        self.data_len = self.idx_descs.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):   
        #print('offset:\n', offset)       
        tree = self.trees[offset]
        #print('tree:\n', type(tree))
        tree_node_num = self.trees_num[offset]
        #print('tree_node_num:\n', tree_node_num)

        if self.training:
            len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
            good_desc_len = min(int(len), self.max_desc_len)
            good_desc = self.descs[pos:pos+good_desc_len]
            good_desc = self.pad_seq(good_desc, self.max_desc_len)
            
            rand_offset = random.randint(0, self.data_len-1)
            len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
            bad_desc_len=min(int(len), self.max_desc_len)
            bad_desc = self.descs[pos:pos+bad_desc_len]
            bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

            return tree, tree_node_num, good_desc, good_desc_len, bad_desc, bad_desc_len
        return tree, tree_node_num, good_desc, good_desc_len
        
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

def load_dict(filename):
    return json.loads(open(filename, "r").readline())
    #return pickle.load(open(filename, 'rb')) 

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs
        
def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()

if __name__ == '__main__':
    device = "cpu"
    input_dir = './data/github12/'
    train_set = CodeSearchDataset(input_dir, 'test.ast.json', 'vocab.ast.json', 'test.desc.h5', 30)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=5, 
                                                    collate_fn=batcher(device), shuffle=False, num_workers=1)
    print('size of train_data_loader:\n', len(train_data_loader))
    '''
    use_set = CodeSearchDataset(input_dir, 'use.tokens.h5', 30)
    use_data_loader = torch.utils.data.DataLoader(dataset=use_set, batch_size=1, shuffle=False, num_workers=1)
    #print(len(use_data_loader))
    vocab_tokens = load_dict(input_dir+'vocab.tokens.json')
    vocab_desc = load_dict(input_dir+'vocab.desc.json')
    '''
    
    print('============ Train Data ================')
    k=0
    for batch in train_data_loader:
        tree, tree_node_num, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        print(tree, tree_node_num)
        k+=1
        if k>0: break
        print('-------------------------------')
        #print(indexes2sent(tokens, vocab_tokens))
        #print(indexes2sent(good_desc, vocab_desc))
    
    
