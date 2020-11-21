
def config_MultiEmbeder():   
    conf = {
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

            # TreeLSTM
            'treelstm_cell_type': 'nary', # nary or childsum
            'n_ast_words': 50000,

            # Token and Description
            'desc_len': 30,
            'tok_len': 100,
            'n_desc_words': 10000, 
            'n_token_words': 25000,

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            # training data
            'train_token':'train.token.h5',
            'train_ast':'train.ast.json',
            'train_cfg':'train.cfg.txt',
            'train_desc':'train.desc.h5',
            # test data
            'test_token':'test.token.h5',
            'test_ast':'test.ast.json',
            'test_cfg':'test.cfg.txt',
            'test_desc':'test.desc.h5', 
            # vocabulary info
            'vocab_token':'vocab.token.json',
            'vocab_ast':'vocab.ast.json',
            'vocab_desc':'vocab.desc.json',
                   
            # model_params
            'emb_size': 300,
            # recurrent  
            'margin': 0.6,
            'sim_measure':'cos',
            'dropout': 0.1,
            
                    
            # training_params            
            'nb_epoch': 200,
            #'optimizer': 'adamW',
            'learning_rate':0.0003, # try 1e-4(paper)
            'adam_epsilon':1e-8,
            'warmup_steps':5000,
            'fp16': False,
            'fp16_opt_level': 'O1' #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                            #"See details at https://nvidia.github.io/apex/amp.html"

        
    }
    return conf

