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

def remove_dup_tokens(args):
    dir_path = args.data_path + args.dataset
    origin_token_file_path = dir_path + args.origin_token_file
    all_token_file_path = dir_path + args.all_token_file

    with open(origin_token_file_path, 'r') as origin_token_file, open(all_token_file_path, 'w') as all_token_file:
        lines = origin_token_file.readlines()
        for i in range(0, len(lines)):
            if lines[i][0:10] != 'BeginFunc:':
                line = lines[i].strip()
                words = line.split()
                new_words = list(set(words))
                new_line = ' '.join(new_words)
                all_token_file.write(new_line + '\n')


def parse_args():
    parser = argparse.ArgumentParser("Parse token data for TokenEmbedder")
    
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='github_user_3/')

    parser.add_argument('--origin_token_file', type=str, default='origin.token.txt')
    parser.add_argument('--all_token_file', type=str, default='all.token.txt')
    parser.add_argument('--train_token_file', type=str, default='train.token.txt')
    parser.add_argument('--test_token_file', type=str, default='test.token.txt')
    parser.add_argument('--vocab_token_file', type=str, default='vocab.token.json')
    
    parser.add_argument('--trainset_num', type=int, default=39152)
    parser.add_argument('--testset_num', type=int, default=2000)
    parser.add_argument('--token_word_num', type=int, default=10000)
    parser.add_argument('--token_maxlen', type=int, default=50)
    parser.add_argument('--testset_start_index', type=int, default=39152)


    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')
 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    '''
    dir_path = args.data_path + args.dataset
    with open(dir_path+'origin.test.token.txt', 'r') as in_file, open(dir_path+'test.token.txt', 'w') as out_file:
        lines = in_file.readlines()
        for i in range(0, len(lines)):
            if lines[i][0:10] != 'BeginFunc:':
                out_file.write(lines[i])
    '''
    remove_dup_tokens(args)
    
    #split_token_data(args)
    #create_token_dict_file(args)

    dir_path = args.data_path + args.dataset
    sents2indexes(dir_path+args.all_token_file, dir_path+args.vocab_token_file, args.token_maxlen)

    '''
    dir_path = args.data_path + args.dataset
    # train.token.txt -> train.token.h5(and test...) 
    sents2indexes(dir_path+args.train_token_file, dir_path+args.vocab_token_file, args.token_maxlen)
    sents2indexes(dir_path+args.test_token_file, dir_path+args.vocab_token_file, args.token_maxlen)
    '''

    '''
    dir_path = args.data_path + args.dataset
    all_token_file_path = dir_path + args.all_token_file
    with open(all_token_file_path, 'r') as all_token_file:
        lines = all_token_file.readlines()
        print(len(lines))
    for i in range(0, len(lines)):
        line = lines[i]
        if line[0:10] != 'BeginFunc:':
            words = line.split()
            if len(words) == 0:
                print(lines[i-1])
                #print(lines[i])
    '''
