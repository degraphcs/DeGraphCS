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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

    
class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """
    def __init__(self, data_dir, f_tokens, max_tok_len, f_descs=None, max_desc_len=None):
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        # initialize file path or list of file names
        """read training data(list of int arrays) from a hdf5 file"""
        self.training = False
        print("loading data...")
        table_tokens = tables.open_file(data_dir+f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]
        if f_descs is not None:
            self.training=True
            table_desc = tables.open_file(data_dir+f_descs)
            self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
            self.idx_descs = table_desc.get_node('/indices')[:]
        
        if f_descs is not None:
            assert self.idx_tokens.shape[0]==self.idx_descs.shape[0]
        self.data_len = self.idx_tokens.shape[0]
        print("{} entries".format(self.data_len))
        
    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq = np.append(seq, [PAD_ID]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq
    
    def __getitem__(self, offset):          
        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos+tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)

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

            return tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len
        return tokens, tok_len, good_desc, good_desc_len
        
    def __len__(self):
        return self.data_len
    

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
    input_dir = './data/github/'
    train_set = CodeSearchDataset(input_dir, 'train.token.h5', 60, 'train.desc.h5', 30)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=False, num_workers=1)
    logger.info('hello')
    #print(len(train_data_loader))
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
        print("batch[0].size(0): ", batch[0].size(0))
        batch = tuple([t.numpy() for t in batch])
        tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
        k+=1
        if k>0: break
        print('-------------------------------')
        #print(indexes2sent(tokens, vocab_tokens))
        #print(indexes2sent(good_desc, vocab_desc))
        
    
