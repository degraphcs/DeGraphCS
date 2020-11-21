
def config_TokenEmbeder():   
    conf = {
            # added_params
            'gpu': 1,
            'transform_every_modal': 0,  # to make modal more complex?
            'save_attn_weight': 0,
            'use_tanh': 1,
            'use_attn': 1,

            # data_params
            'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            #training data
            'train_tokens':'train.token.h5',
            'train_desc':'train.desc.h5',
            #valid data
            'valid_tokens':'valid.token.h5',
            'valid_desc':'valid.desc.h5',
            # test data
            'test_tokens':'test.token.h5',
            'test_desc':'test.desc.h5', 
                   
            #parameters
            'tokens_len':50,
            'desc_len': 30,
            'n_token_words': 20000, # len(vocabulary) + 1 
            'n_desc_words': 12000, # wait to decide
            #vocabulary info
            'vocab_tokens':'vocab.token.json',
            'vocab_desc':'vocab.desc.json',
                    
        #training_params            
            'batch_size': 32,
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
            'n_hidden': 512,#number of hidden dimension of code/desc representation
            # recurrent  
            'margin': 0.6,
            'sim_measure':'cos',#similarity measure: cos, poly, sigmoid, euc, gesd, aesd. see https://arxiv.org/pdf/1508.01585.pdf
                         #cos, poly and sigmoid are fast with simple dot, while euc, gesd and aesd are slow with vector normalization.
            'dropout':0.1
    }
    return conf

