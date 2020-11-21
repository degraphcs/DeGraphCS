import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import random
import time
from datetime import datetime
import numpy as np
import math
import argparse
random.seed(42)
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

import models, configs 
from modules import get_cosine_schedule_with_warmup
from data_loader import *

    
def train(args):
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logs.txt")
                                      # create file handler which logs even debug messages
    logger.addHandler(fh)# add the handlers to the logger
    timestamp = datetime.now().strftime('%Y%m%d%H%M') 
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu") 
    
    def save_model(model, epoch):
        torch.save(model.state_dict(), f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5')

    def load_model(model, epoch, to_device):
        assert os.path.exists(f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5'), f'Weights at epoch {epoch} not found'
        model.load_state_dict(torch.load(f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5', map_location=to_device))

    config = getattr(configs, 'config_'+args.model)()
    print(config)
    
    # load data
    data_path = args.data_path+args.dataset+'/'
    train_set = eval(config['dataset_name'])(config, data_path, 
                                config['train_token'], config['tok_len'],
                                config['train_ast'], config['vocab_ast'],
                                config['train_cfg'], config['n_node'],
                                config['train_desc'], config['desc_len'])
    
    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], 
                                       collate_fn=batcher(device), shuffle=True, drop_last=False, num_workers=0)
    
    # define the models
    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config) #initialize the model
    if args.reload_from>0:
        load_model(model, args.reload_from, device)    
    logger.info('done')
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])        
    scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=config['warmup_steps'], 
            num_training_steps=len(data_loader)*config['nb_epoch']) # do not forget to modify the number when dataset is changed

    print('---model parameters---')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    n_iters = len(data_loader)
    itr_global = args.reload_from+1 
    for epoch in range(int(args.reload_from)+1, config['nb_epoch']+1): 
        itr_start_time = time.time()
        losses=[]
        for batch in data_loader:
            
            model.train()
            batch_gpu = [tensor for tensor in batch]
            loss = model(*batch_gpu)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            losses.append(loss.item())
            
            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f'%
                        (epoch, config['nb_epoch'], itr_global%n_iters, n_iters, elapsed, np.mean(losses)))
                    
                losses=[] 
                itr_start_time = time.time() 
            itr_global = itr_global + 1

        # save every epoch
        if epoch >= 90:
            if epoch % 5 == 0:
                save_model(model, epoch)

    
def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='MultiEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='github', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')
   
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual",action="store_true", default=False, help="Visualize training status in tensorboard")
    # Training Arguments
    parser.add_argument('--log_every', type=int, default=50, help='interval to log autoencoder training results')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
        
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # make output directory if it doesn't already exist
    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)
    
    torch.backends.cudnn.benchmark = True # speed up training by using cudnn
    torch.backends.cudnn.deterministic = True # fix the random seed in cudnn
   
    train(args)
        