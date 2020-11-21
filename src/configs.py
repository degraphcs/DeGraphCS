
def config_IREmbeder():   
    conf = {
            # added_params
            'transform_every_modal': 0,
            'use_attn': 0,
            'use_tanh': 1,
            'save_attn_weight': 0,

            # GGNN
            'state_dim': 512, # GGNN hidden state size
            'annotation_dim': 300,
            'n_edge_types': 2,
            'n_node': 160, # maximum nodenum
            'n_steps': 5, # propogation steps number of GGNN
            'output_type': 'no_reduce',
            'batch_size': 32,
            'n_layers': 1,
            'n_hidden': 512,
            'ir_attn_mode': 'sigmoid_scalar',
            'word_split': True,
            'pooling_type': 'max_pooling', # ave_pooling
            'max_word_num': 5,

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            # training data
            'train_ir':'train.ir.json',
            'train_desc':'train.desc.h5',
            # test data
            'test_ir':'test.ir.json',
            'test_desc':'test.desc.h5', 
                   
            # parameters
            'desc_len': 30,
            'n_desc_words': 10000, 
            'n_ir_words': 15000,
            # vocabulary info
            'vocab_ir':'vocab.ir.json',
            'vocab_desc':'vocab.desc.json',
                    
            #training_params            
            'nb_epoch': 100,
            #'optimizer': 'adam',
            'learning_rate':0.0003, # try 1e-4(paper)
            'adam_epsilon':1e-8,
            'warmup_steps':5000,
            'fp16': False,
            'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].

            # model_params
            'emb_size': 300,
            # recurrent  
            'margin': 0.6,
            'sim_measure':'cos',
            'dropout': 0
    }
    return conf

