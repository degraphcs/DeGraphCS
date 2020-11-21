import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

import models, configs, data_loader 
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *


def test(config, model, device):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    
    data_path = args.data_path+args.dataset+'/'

    code_base_set = eval(config['dataset_name'])(data_path,
                                  config['all_name'], config['name_len'],
                                  config['all_tokens'], config['tokens_len'])
    code_data_loader = torch.utils.data.DataLoader(dataset=code_base_set, batch_size=32,
                                        shuffle=False, drop_last=False, num_workers=1)
    
    code_reprs = []
    code_processed = 0
    for batch in code_data_loader:
        # batch[0:4]: name, name_len, token, token_len
        code_batch = [tensor.to(device) for tensor in batch[:4]]   
        with torch.no_grad():
            code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
            code_repr = normalize(code_repr)   
        code_reprs.append(code_repr)
        code_processed += batch[0].size(0) # +batch_size
    # code_reprs: [code_processed x n_hidden]
    code_reprs = np.vstack(code_reprs)
    print('processed code num: ', code_processed)
    
    
    query_desc_set = eval(config['dataset_name'])(data_path,
                                f_descs=config['query_desc'], max_desc_len=config['desc_len'])
    desc_data_loader = torch.utils.data.DataLoader(dataset=query_desc_set, batch_size=32,
                                        shuffle=False, drop_last=False, num_workers=1)
    
    desc_reprs = []
    desc_processed = 0
    for batch in desc_data_loader:
        # batch[0:2]: good_desc, good_desc_len
        desc_batch = [tensor.to(device) for tensor in batch[0:2]]
        with torch.no_grad():
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32) # [poolsize x hidden_size]
            desc_repr = normalize(desc_repr)
        desc_reprs.append(desc_repr)
        desc_processed += batch[0].size(0) # +batch_size
    # desc_reprs: [desc_processed x n_hidden]
    desc_reprs = np.vstack(desc_reprs)
    print('processed desc num: ', desc_processed)

    
    query_desc_index_file_path = data_path + args.query_desc_index_file
    desc_index = []
    with open(query_desc_index_file_path, 'r') as query_desc_index_file:
        lines = query_desc_index_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i].strip()
            desc_index.append(int(line))
    print('desc_index: ', desc_index)

    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []
    test_sim_result, test_rank_result = [], []
    for i in tqdm(range(0, desc_processed)):
        ind = desc_index[i]

        desc_vec = np.expand_dims(desc_reprs[i], axis=0) # [1 x n_hidden]
        sims = np.dot(code_reprs, desc_vec.T)[:, 0] # [code_processed]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)
        
        # SuccessRate@k
        predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in predict[0:10]]
        sum_1.append(1.0) if ind in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if ind in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if ind in predict_10 else sum_10.append(0.0)
        # MRR
        predict_list = predict.tolist()
        rank = predict_list.index(ind)
        sum_mrr.append(1/float(rank+1))

        # results need to be saved
        predict_20 = [int(k) for k in predict[0:20]]
        sim_20 = [sims[k] for k in predict_20]
        test_sim_result.append(zip(predict_20, sim_20))
        test_rank_result.append(rank+1)

    logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')
    save_path = args.data_path + 'result/'
    sim_result_filename, rank_result_filename = 'sim.npy', 'rank.npy'
    np.save(save_path+sim_result_filename, test_sim_result)
    np.save(save_path+rank_result_filename, test_rank_result)
    
    
def parse_args():
    parser = argparse.ArgumentParser("Test Code Search(Embedding) Model For User Study")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='github_user_3', help='name of dataset.java, python')
    parser.add_argument('--query_desc_index_file', type=str, default='query.desc.index.txt')
    parser.add_argument('--reload_from', type=int, default=185, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    config = getattr(configs, 'config_'+args.model)()
    
    ##### Define model ######
    logger.info('Constructing Model..')
    
    model = getattr(models, args.model)(config) # initialize the model
    ckpt=f'./output/{args.model}/{args.dataset}/models/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    test(config, model, device)    


