
def config_CFGEmbeder():   
    conf = {
            # added_params
            'transform_every_modal': 0,  # to make modal more complex?
            'save_attn_weight': 0,
            'use_tanh': 1,
            'use_attn': 1,

            # GGNN
            'state_dim': 512, # GGNN hidden state size
            'annotation_dim': 5,
            'n_edge_types': 2,
            'n_node': 200, # could be less than 512, like the maximum nodenum
            'n_steps': 5, # propogation steps number of GGNN
            'output_type': 'no_reduce',
            'batch_size': 32,
            'n_layers': 1,
            'n_hidden': 512,
            'cfg_attn_mode': 'sigmoid_scalar',

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            #training data
            'train_cfg':'train.cfg.txt',
            'train_desc':'train.desc.h5',
            # test data
            'test_cfg':'test.cfg.txt',
            'test_desc':'test.desc.h5', 
                   
            #parameters
            'desc_len': 30,
            'n_desc_words': 10000, # wait to decide
            #vocabulary info
            'vocab_desc':'vocab.desc.json',
                    
            #training_params            
            'chunk_size': 200000,
            'nb_epoch': 200,
            #'optimizer': 'adam',
            'learning_rate':0.0003, # try 1e-4(paper)
            'adam_epsilon':1e-8,
            'warmup_steps':5000,
            'fp16': False,
            'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                            #"See details at https://nvidia.github.io/apex/amp.html"

        # model_params
            'emb_size': 300,
            # recurrent  
            'margin': 0.6,
            'sim_measure':'cos',#similarity measure: cos, poly, sigmoid, euc, gesd, aesd. see https://arxiv.org/pdf/1508.01585.pdf
                         #cos, poly and sigmoid are fast with simple dot, while euc, gesd and aesd are slow with vector normalization.
            'dropout': 0.1
    }
    return conf

