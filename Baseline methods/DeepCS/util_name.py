import numpy as np
import argparse
from collections import Counter
import json
import h5py

PAD_ID, UNK_ID = [0, 1]

def split_name_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_name_file_path = dir_path + args.all_name_file
    train_name_file_path = dir_path + args.train_name_file
    test_name_file_path = dir_path + args.test_name_file

    input_name = []
    with open(all_name_file_path, 'r') as all_name_file:
        lines = all_name_file.readlines()
        for line in lines:
            if (line[0:10] != 'BeginFunc:'):
                input_name.append(line)
        print('number of input name:\n', len(input_name))

    with open(train_name_file_path, 'w') as train_name_file, open(test_name_file_path, 'w') as test_name_file:
        for i in range(0, args.trainset_num):
            train_name_file.write(input_name[index[i]])
        for i in range(args.testset_start_index, args.testset_start_index+args.testset_num):
            test_name_file.write(input_name[index[i]])
    

def create_name_dict_file(args):
    dir_path = args.data_path + args.dataset
    name_file_path = dir_path + args.train_name_file 

    input_name = []
    with open(name_file_path, 'r') as name_file:
        input_name = name_file.readlines()
    name_words = []
    for i in range(0, len(input_name)):
        input_name[i] = input_name[i].rstrip('\n')
        name_word_list = input_name[i].split()
        for name_word in name_word_list:
            name_words.append(name_word)
    vocab_name_info = Counter(name_words)
    print(len(vocab_name_info))

    vocab_name = [item[0] for item in vocab_name_info.most_common()[:args.name_word_num-2]]
    vocab_name_index = {'<pad>':0, '<unk>':1}
    vocab_name_index.update(zip(vocab_name, [item+2 for item in range(len(vocab_name))]))

    
    vocab_name_file_path = dir_path + args.vocab_name_file
    name_dic_str = json.dumps(vocab_name_index)
    with open(vocab_name_file_path, 'w') as vocab_name_file:
        vocab_name_file.write(name_dic_str)

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

def create_name_file(args):
    dir_path = args.data_path + args.dataset
    origin_name_file_path = dir_path + args.origin_name_file
    all_name_file_path = dir_path + args.all_name_file

    with open(origin_name_file_path, 'r') as origin_name_file, open(all_name_file_path, 'w') as all_name_file:
        lines = origin_name_file.readlines()
        for i in range(0, len(lines)):
            if lines[i][0:10] == 'BeginFunc:':
                #all_name_file.write(lines[i])
                line = lines[i].strip()
                ind = line.rfind(':')+1
                name = clean_word(line[ind:])
                all_name_file.write(name + '\n')

def clean_word(word_input):
    word = ''
    for i in range(0, len(word_input)):
        if word_input[i] >= 'A' and word_input[i] <= 'Z':
            word += '_'
            word += word_input[i]
        else:
            word += word_input[i]
    word = word.lower()
    if '_' not in word:
        return word
    if word[0] == '_':
        new_word = ''
    else:       
        new_word = word[0]
    for i in range(1, len(word)):
        if word[i] == '_' and word[i-1] == '_':
            continue
        elif word[i] == '_':
            new_word = new_word + ' '
        else:
            new_word = new_word + word[i]
    return new_word



def parse_args():
    parser = argparse.ArgumentParser("Parse name data for nameEmbedder")
    
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='github_user_3/')

    parser.add_argument('--origin_name_file', type=str, default='origin.token.txt')
    parser.add_argument('--all_name_file', type=str, default='all.name.txt')
    parser.add_argument('--train_name_file', type=str, default='train.name.txt')
    parser.add_argument('--test_name_file', type=str, default='test.name.txt')
    parser.add_argument('--vocab_name_file', type=str, default='vocab.name.json')
    
    parser.add_argument('--trainset_num', type=int, default=39152)
    parser.add_argument('--testset_num', type=int, default=2000)
    parser.add_argument('--name_word_num', type=int, default=10000)
    parser.add_argument('--name_maxlen', type=int, default=6)
    parser.add_argument('--testset_start_index', type=int, default=39152)


    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')
 
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    '''
    dir_path = args.data_path + args.dataset
    with open(dir_path+'origin.test.name.txt', 'r') as in_file, open(dir_path+'test.name.txt', 'w') as out_file:
        lines = in_file.readlines()
        for i in range(0, len(lines)):
            if lines[i][0:10] != 'BeginFunc:':
                out_file.write(lines[i])
    '''

    create_name_file(args)
    
    #split_name_data(args)
    #create_name_dict_file(args)
    '''
    dir_path = args.data_path + args.dataset
    # train.name.txt -> train.name.h5(and test...) 
    sents2indexes(dir_path+args.train_name_file, dir_path+args.vocab_name_file, args.name_maxlen)
    sents2indexes(dir_path+args.test_name_file, dir_path+args.vocab_name_file, args.name_maxlen)
    '''

    dir_path = args.data_path + args.dataset
    sents2indexes(dir_path+args.all_name_file, dir_path+args.vocab_name_file, args.name_maxlen)

    '''
    dir_path = args.data_path + args.dataset
    all_name_file_path = dir_path + args.all_name_file
    with open(all_name_file_path, 'r') as all_name_file:
        lines = all_name_file.readlines()
        print(len(lines))
    for i in range(0, len(lines)):
        line = lines[i]
        if line[0:10] != 'BeginFunc:':
            words = line.split()
            if len(words) == 0:
                print(lines[i-1])
                #print(lines[i])
    '''
