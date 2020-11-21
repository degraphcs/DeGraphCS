import numpy as np
import argparse
from collections import Counter
import json
import h5py

PAD_ID, UNK_ID = [0, 1]

def make_shuffle_index(args):
    dir_path = args.data_path + args.dataset
    all_desc_file_path = dir_path + args.all_desc_file
    with open(all_desc_file_path, 'r') as all_desc_file:
        lines = all_desc_file.readlines()
        all_num = int(len(lines)/2) 

    index = np.arange(all_num)
    np.random.seed(16)
    np.random.shuffle(index)
    print(len(index))
    np.save(args.shuffle_index_file, index)

def split_desc_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_desc_file_path = dir_path + args.all_desc_file
    train_desc_file_path = dir_path + args.train_desc_file
    test_desc_file_path = dir_path + args.test_desc_file

    input_desc = []
    with open(all_desc_file_path, 'r') as all_desc_file:
        lines = all_desc_file.readlines()
        for line in lines:
            if (line[0:10] != 'BeginFunc:'):
                input_desc.append(line)
        print('number of input desc:\n', len(input_desc))

    with open(train_desc_file_path, 'w') as train_desc_file, open(test_desc_file_path, 'w') as test_desc_file:
        for i in range(0, args.trainset_num):
            train_desc_file.write(input_desc[index[i]])
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            test_desc_file.write(input_desc[index[i]])
    

def create_desc_dict_file(args):
    dir_path = args.data_path + args.dataset
    desc_file_path = dir_path + args.train_desc_file 

    input_desc = []
    with open(desc_file_path, 'r') as desc_file:
        input_desc = desc_file.readlines()
    desc_words = []
    for i in range(0, len(input_desc)):
        input_desc[i] = input_desc[i].rstrip('\n')
        desc_word_list = input_desc[i].split()
        for desc_word in desc_word_list:
            desc_words.append(desc_word)
    vocab_desc_info = Counter(desc_words)
    print(len(vocab_desc_info))

    vocab_desc = [item[0] for item in vocab_desc_info.most_common()[:args.desc_word_num-2]]
    vocab_desc_index = {'<pad>':0, '<unk>':1}
    vocab_desc_index.update(zip(vocab_desc, [item+2 for item in range(len(vocab_desc))]))

    
    vocab_desc_file_path = dir_path + args.vocab_desc_file
    desc_dic_str = json.dumps(vocab_desc_index)
    with open(vocab_desc_file_path, 'w') as vocab_desc_file:
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
    parser = argparse.ArgumentParser("Parse Description data for TokenEmbedder")
    
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='github11/')

    parser.add_argument('--origin_desc_file', type=str, default='origin.desc.txt')
    parser.add_argument('--all_desc_file', type=str, default='all.desc.txt')
    parser.add_argument('--train_desc_file', type=str, default='train.desc.txt')
    parser.add_argument('--test_desc_file', type=str, default='test.desc.txt')
    parser.add_argument('--vocab_desc_file', type=str, default='vocab.desc.json')
    
    parser.add_argument('--trainset_num', type=int, default=39152)
    parser.add_argument('--testset_num', type=int, default=2000)
    parser.add_argument('--desc_word_num', type=int, default=10000)
    parser.add_argument('--desc_maxlen', type=int, default=30)
    parser.add_argument('--testset_start_index', type=int, default=39152)

    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')
 
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    #make_shuffle_index(args)
    
    #split_desc_data(args)
    
    create_desc_dict_file(args)

    dir_path = args.data_path + args.dataset
    # train.desc.txt -> train.desc.h5(and test...) 
    sents2indexes(dir_path+args.train_desc_file, dir_path+args.vocab_desc_file, args.desc_maxlen)
    sents2indexes(dir_path+args.test_desc_file, dir_path+args.vocab_desc_file, args.desc_maxlen)
    
