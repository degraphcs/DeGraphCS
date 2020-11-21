import os
import sys
import torch
import numpy as np
import argparse
import pickle
from collections import Counter
from utils import PAD_ID, UNK_ID, indexes2sent
import json
import h5py
import tables

import nltk
try: nltk.word_tokenize("hello world")
except LookupError: nltk.download('punkt')

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def split_data(args):
    # read file of needed length
    dir_path = args.data_path + args.dataset
    token_file_path = dir_path + args.token_file
    desc_file_path = dir_path + args.desc_file
    total_num = args.trainset_num + args.validset_num + args.testset_num
    with open(token_file_path, 'r') as token_file, open(desc_file_path, 'r') as desc_file:
        input_token_origin = token_file.readlines()
        input_desc_origin = desc_file.readlines()
    if len(input_token_origin) != len(input_desc_origin):
        assert False, ('token and desc are not corresponded!')
    for i in range(0, len(input_token_origin)):
        input_token_origin[i] = input_token_origin[i].rstrip('\n')
        input_desc_origin[i] = input_desc_origin[i].rstrip('\n')
    
    index = np.load(args.shuffle_index_file)
    input_token, input_desc = [], []
    for i in range(0, len(index)):
        input_token.append(input_token_origin[index[i]])
        input_desc.append(input_desc_origin[index[i]])
    save_token_file_path = dir_path + args.save_token_file
    save_desc_file_path = dir_path + args.save_desc_file

    # split data for training, validation, test
    train_token_file_path = dir_path + args.train_token_file
    test_token_file_path = dir_path + args.test_token_file
    train_desc_file_path = dir_path + args.train_desc_file
    test_desc_file_path = dir_path + args.test_desc_file
    start = 0
    with open(train_token_file_path, 'w') as train_token_file, open(train_desc_file_path, 'w') as train_desc_file:
        for i in range(0, args.trainset_num):
            train_token_file.write(input_token[i]+'\n')
            train_desc_file.write(input_desc[i]+'\n')
    start += args.trainset_num
    with open(test_token_file_path, 'w') as test_token_file, open(test_desc_file_path, 'w') as test_desc_file:
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            test_token_file.write(input_token[i]+'\n')
            test_desc_file.write(input_desc[i]+'\n')


def create_dict_file(args):
    dir_path = args.data_path + args.dataset
    train_token_file_path = dir_path + args.train_token_file
    train_desc_file_path = dir_path + args.train_desc_file
    input_token, input_desc = [], []
    with open(save_token_file_path, 'r') as token_file, open(save_desc_file_path, 'r') as desc_file:
        input_token = token_file.readlines()
        input_desc = desc_file.readlines()
    token_words, desc_words = [], []
    for i in range(0, len(input_token)):
        input_token[i] = input_token[i].rstrip('\n')
        token_word_list = input_token[i].split()
        for token_word in token_word_list:
            token_words.append(token_word)
        input_desc[i] = input_desc[i].rstrip('\n')
        desc_word_list = input_desc[i].split()
        for desc_word in desc_word_list:
            desc_words.append(desc_word)
    vocab_token_info = Counter(token_words)
    vocab_desc_info = Counter(desc_words)
    print(len(vocab_token_info))
    print(len(vocab_desc_info))
    vocab_token = [item[0] for item in vocab_token_info.most_common()[:args.token_word_num-2]]
    vocab_desc = [item[0] for item in vocab_desc_info.most_common()[:args.desc_word_num-2]]
    vocab_token_index = {'<pad>':0, '<unk>':1}
    vocab_desc_index = {'<pad>':0, '<unk>':1}
    vocab_token_index.update(zip(vocab_token, [item+2 for item in range(len(vocab_token))]))
    vocab_desc_index.update(zip(vocab_desc, [item+2 for item in range(len(vocab_desc))]))

    # save dict file
    vocab_token_file_path = dir_path + args.vocab_token_file
    vocab_desc_file_path = dir_path + args.vocab_desc_file
    token_dic_str = json.dumps(vocab_token_index)
    desc_dic_str = json.dumps(vocab_desc_index)
    with open(vocab_token_file_path,'w') as vocab_token_file, open(vocab_desc_file_path, 'w') as vocab_desc_file:
        vocab_token_file.write(token_dic_str)
        vocab_desc_file.write(desc_dic_str)


def sents2indexes(sent_file_path, vocab_file_path, maxlen):
    phrases, indices = [], []
    with open(sent_file_path, 'r') as sent_file:
        sents = sent_file.readlines()
    vocab = json.loads(open(vocab_file_path, "r").readline())
    start_index = 0
    for i in range(0, len(sents)):
        sent = sents[i].rstrip('\n')
        word_list = sent.split()
        sent_len = min(len(word_list), maxlen)
        indices.append((sent_len, start_index))
        for j in range(0, sent_len):
            word = word_list[j]
            phrases.append(vocab.get(word, UNK_ID))
        start_index += sent_len
    output_file_path = sent_file_path[0:-3] + 'h5'
    output_file = h5py.File(output_file_path, 'w')
    output_file['phrases'] = phrases
    output_file['indices'] = indices
    output_file.close()


def parse_args():
    parser = argparse.ArgumentParser("Split Dataset, Create Dict and Generate Files")
    
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='github/', help='name of dataset.c')
    parser.add_argument('--token_file', type=str, default='all.token.txt')
    parser.add_argument('--desc_file', type=str, default='all.desc.txt')

    parser.add_argument('--train_token_file', type=str, default='train.token.txt')
    parser.add_argument('--test_token_file', type=str, default='test.token.txt')
    parser.add_argument('--train_desc_file', type=str, default='train.desc.txt')
    parser.add_argument('--test_desc_file', type=str, default='test.desc.txt')
    parser.add_argument('--vocab_token_file', type=str, default='vocab.token.json')
    parser.add_argument('--vocab_desc_file', type=str, default='vocab.desc.json')

    parser.add_argument('--trainset_num', type=int, default=32000, help='length of training set')
    parser.add_argument('--testset_num', type=int, default=1000, help='length of test set')
    parser.add_argument('--token_word_num', type=int, default=20000)
    parser.add_argument('--desc_word_num', type=int, default=10000)
    parser.add_argument('--token_maxlen', type=int, default=50)
    parser.add_argument('--desc_maxlen', type=int, default=30)
    parser.add_argument('--testset_start_index', type=int, default=33000)


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    split_data(args)
    create_dict_file(args)

    dir_path = args.data_path + args.dataset
    token_file_name = [args.train_token_file, args.test_token_file]
    desc_file_name = [args.train_desc_file, args.test_desc_file]
    for i in range(0, lenn(token_file_name)):
        sents2indexes(dir_path+token_file_name[i], args.vocab_token_file, args.token_maxlen)
        sents2indexes(dir_path+desc_file_name[i], args.vocab_desc_file, args.desc_maxlen)
       


