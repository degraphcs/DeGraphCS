import numpy as np
import argparse
from collections import Counter
import json
import h5py

PAD_ID, UNK_ID = [0, 1]

def split_token_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_token_file_path = dir_path + args.all_token_file
    train_token_file_path = dir_path + args.train_token_file
    test_token_file_path = dir_path + args.test_token_file

    input_token = []
    with open(all_token_file_path, 'r') as all_token_file:
        lines = all_token_file.readlines()
        for line in lines:
            if (line[0:10] != 'BeginFunc:'):
                input_token.append(line)
        print('number of input token:\n', len(input_token))

    with open(train_token_file_path, 'w') as train_token_file, open(test_token_file_path, 'w') as test_token_file:
        for i in range(0, args.trainset_num):
            train_token_file.write(input_token[index[i]])
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            test_token_file.write(input_token[index[i]])
    

def create_token_dict_file(args):
    dir_path = args.data_path + args.dataset
    token_file_path = dir_path + args.train_token_file 

    input_token = []
    with open(token_file_path, 'r') as token_file:
        input_token = token_file.readlines()
    token_words = []
    for i in range(0, len(input_token)):
        input_token[i] = input_token[i].rstrip('\n')
        token_word_list = input_token[i].split()
        for token_word in token_word_list:
            token_words.append(token_word)
    vocab_token_info = Counter(token_words)
    print(len(vocab_token_info))

    vocab_token = [item[0] for item in vocab_token_info.most_common()[:args.token_word_num-2]]
    vocab_token_index = {'<pad>':0, '<unk>':1}
    vocab_token_index.update(zip(vocab_token, [item+2 for item in range(len(vocab_token))]))

    
    vocab_token_file_path = dir_path + args.vocab_token_file
    token_dic_str = json.dumps(vocab_token_index)
    with open(vocab_token_file_path, 'w') as vocab_token_file:
        vocab_token_file.write(token_dic_str)


'''
def parse_args():
    parser = argparse.ArgumentParser("Parse tokenription data for CFGEmbedder")
    
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='example/')

    parser.add_argument('--all_token_file', type=str, default='all.token.txt')
    parser.add_argument('--train_token_file', type=str, default='train.token.txt')
    parser.add_argument('--test_token_file', type=str, default='test.token.txt')
    parser.add_argument('--vocab_token_file', type=str, default='vocab.token.json')
    
    parser.add_argument('--trainset_num', type=int, default=12)
    parser.add_argument('--testset_num', type=int, default=1000)
    parser.add_argument('--token_word_num', type=int, default=50)
    parser.add_argument('--token_maxlen', type=int, default=50)
    parser.add_argument('--testset_start_index', type=int, default=33000)


    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')
 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    #make_shuffle_index(args)
    #split_data(args)
    #create_token_dict_file(args)

    
    #dir_path = args.data_path + args.dataset
    # train.token.txt -> train.token.h5(and test...) 
    #sents2indexes(dir_path+args.train_token_file, dir_path+args.vocab_token_file, args.token_maxlen)
    #sents2indexes(dir_path+args.test_token_file, dir_path+args.vocab_token_file, args.token_maxlen)
'''
