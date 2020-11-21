import numpy as np
import configs
import argparse
from collections import Counter
import json
from utils import PAD_ID, UNK_ID



def get_one_ir_npy_info(json_graph_dict, n_node, n_edge_types, max_word_num):
    
    node_num = min(len(json_graph_dict), n_node)
    save_edge_digit_list = []
    word_num = []
    anno = np.zeros([n_node, max_word_num])
    
    for i in range(0, node_num):
        word_list = json_graph_dict[str(i)]['wordid']
        word_num_this_node = len(word_list)
        
        for j in range(0, word_num_this_node):
            anno[i][j] = word_list[j]

        is_control_edge = 0
        if word_num_this_node == 1 and word_list[0] == 2: 
            is_control_edge = 1

        if 'snode' in json_graph_dict[str(i)].keys():
            snode_list = json_graph_dict[str(i)]['snode']
            for j in range(0, len(snode_list)):
                snode = snode_list[j]
                if snode < n_node: 
                    if is_control_edge == 1:
                        save_edge_digit_list.append([i, snode, 1]) 
                    else:
                        word_list = json_graph_dict[str(snode)]['wordid']
                        if len(word_list) == 1 and word_list[0] == 2:
                            save_edge_digit_list.append([i, snode, 1])
                        else:
                            save_edge_digit_list.append([i, snode, 0])
                    
    
    adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
    
    node_mask = [1 if k < node_num else 0 for k in range(0, n_node)]

    return anno, adjmat, node_mask


def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        tgt_idx = edge[1]
        e_type = edge[2]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a
    

def split_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file
    train_ir_file_path = dir_path + args.train_ir_file
    test_ir_file_path = dir_path + args.test_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        lines = all_ir_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('all_num of ir:\n', len(mark_list))

    with open(train_ir_file_path, 'w') as train_ir_file,  open(test_ir_file_path, 'w') as test_ir_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_ir_file.write(lines[j])
        for i in range(args.testset_start_ind, args.testset_start_ind+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_ir_file.write(lines[j])


def transform_edge_to_node(args):
    dir_path = args.data_path + args.dataset
    origin_ir_file_path = dir_path + args.origin_ir_file
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(origin_ir_file_path, 'r') as origin_ir_file:
        ir_lines = origin_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    with open(all_ir_file_path, 'w') as all_ir_file:
        for i in range(0, len(mark_list)):
            s_ind = mark_list[i][0]
            e_ind = mark_list[i][1]
            all_ir_file.write(ir_lines[s_ind])
    
            ir_graph_info_list = ir_lines[s_ind+1].split()
            node_num = int(ir_graph_info_list[0])
            edge_num = int(ir_graph_info_list[1])
            all_ir_file.write(str(node_num+edge_num) + ' ' + ir_graph_info_list[1] + '\n')

            for j in range(s_ind+2, e_ind):
                line = ir_lines[j]
                edge_info_list = line.split()

                if len(edge_info_list) == 2:
                    all_ir_file.write(line)
                else:
                    all_ir_file.write(edge_info_list[0] + ' ' + str(node_num) + ':' + edge_info_list[2] + '\n')
                    all_ir_file.write(str(node_num) + ':' + edge_info_list[2] + ' ' + edge_info_list[1] + '\n')
                    node_num += 1



def observe_data(args):
    dir_path = args.data_path + args.dataset
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(all_ir_file_path, 'r') as all_ir_file:
        ir_lines = all_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    max_word_num = 0
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]
        e_ind = mark_list[i][1]

        for j in range(s_ind+2, e_ind):
            line = ir_lines[j]
            edge_info_list = line.split()

            start_node_list = edge_info_list[0].split(':')
            end_node_list = edge_info_list[1].split(':')
            s_word = start_node_list[1]
            len1 = len(s_word.split('_'))
            e_word = end_node_list[1]
            len2 = len(e_word.split('_'))
            if len1 > max_word_num:
                max_word_num = len1
                print(s_word)
            if len2 > max_word_num:
                max_word_num = len2
                print(e_word)
    print(max_word_num)



def preprocess_origin_ir(args):
    dir_path = args.data_path + args.dataset
    origin_ir_file_path = dir_path + args.origin_ir_file
    all_ir_file_path = dir_path + args.all_ir_file

    mark_list = []
    start_index, end_index = [0, 0]
    with open(origin_ir_file_path, 'r') as origin_ir_file:
        ir_lines = origin_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])

    with open(all_ir_file_path, 'w') as all_ir_file:
        for i in range(0, len(mark_list)):
            s_ind = mark_list[i][0]
            e_ind = mark_list[i][1]
            all_ir_file.write(ir_lines[s_ind])
            all_ir_file.write(ir_lines[s_ind+1])

            for j in range(s_ind+2, e_ind):
                line = ir_lines[j]
                edge_info_list = line.split()

                start_node_list = edge_info_list[0].split(':')
                s_node = start_node_list[1]

                end_node_list = edge_info_list[1].split(':')
                e_node = end_node_list[1]
                
                all_ir_file.write(start_node_list[0]+':'+clean_node(s_node)+' '+
                                    end_node_list[0]+':'+clean_node(e_node)+'\n')


        
def clean_node(node_str):
    
    if node_str.isdigit():
        return 'ID'

    
    if node_str[0:9] == '__func__.':
        node_str = 'func_' + node_str[9:]
    if node_str[0:13] == '__FUNCTION__.':
        node_str = 'function_' + node_str[13:]
    if node_str[0:6] == 'FLAC__':
        node_str = 'flac_' + node_str[6:]
    
    new_node_str = ''
    for i in range(0, len(node_str)):
        if node_str[i] == '.':
            new_node_str += '_'
        elif node_str[i] >= '0' and node_str[i] <= '9':
            continue
        elif node_str[i] >= 'A' and node_str[i]  <= 'Z':
            new_node_str += node_str[i].lower()
        else:
            new_node_str += node_str[i]
    #print(new_node_str)
    
    if len(new_node_str) == 1:
        return new_node_str
    
    
    new2_node_str = ''
    for i in range(0, len(new_node_str)):
        if i == 0:
            if new_node_str[i+1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]
        elif i == len(new_node_str)-1:
            if new_node_str[i-1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]
        else:
            if new_node_str[i-1] == '_' and new_node_str[i+1] == '_':
                new2_node_str += '_'
            else:
                new2_node_str += new_node_str[i]

    flag = 0
    for i in range(0, len(new2_node_str)):
        if new2_node_str[i] != '_':
            flag = 1
    if flag == 0:
        new2_node_str = new_node_str
    
    new2_node_str = new2_node_str.strip('_')
    new3_node_str = ''
    for i in range(0, len(new2_node_str)):
        if i == len(new2_node_str)-1:
            new3_node_str += new2_node_str[i]
        elif new2_node_str[i] == '_' and new2_node_str[i+1] == '_':
            continue
        else:
            new3_node_str += new2_node_str[i]
    
    cnt_num = 0
    for i in range(0, len(new3_node_str)):
        if new3_node_str[i] == '_':
            cnt_num += 1
        if cnt_num == 5:
            new3_node_str = new3_node_str[0:i]
            break
    

    return new3_node_str
        


def create_dict_file(args):
    dir_path = args.data_path + args.dataset
    train_ir_file_path = dir_path + args.train_ir_file

    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()

    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)])    

    ir_words = []
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]
        e_ind = mark_list[i][1]
        for j in range(s_ind+2, e_ind):
            edge_info_list = ir_lines[j].split()
            s_node_list = edge_info_list[0].split(':')
            e_node_list = edge_info_list[1].split(':')

            
            if args.word_split_type == 'split':
                if s_node_list[1] == 'control_label' or s_node_list[1] == 'return_point':
                    ir_words.append(s_node_list[1])
                else:
                    s_node = s_node_list[1].split('_')
                    for i in range(0, len(s_node)):
                        ir_words.append(s_node[i])
                if e_node_list[1] == 'control_label' or e_node_list[1] == 'return_point':
                    ir_words.append(e_node_list[1])
                else:
                    e_node = e_node_list[1].split('_')
                    for i in range(0, len(e_node)):
                        ir_words.append(e_node[i])
        
            else:
                ir_words.append(s_node_list[1])
                ir_words.append(e_node_list[1])

    vocab_ir_info = Counter(ir_words)
    print(len(vocab_ir_info))
    print(vocab_ir_info)
    '''
    tmp = vocab_ir_info.most_common()
    print(tmp[25000])
    for i in range(0, len(tmp)):
        t = tmp[i]
        if (t[1] == 4):
            print(i)
            break
    '''
    vocab_ir = [item[0] for item in vocab_ir_info.most_common()[:args.ir_word_num-2]]
    vocab_ir_index = {'<pad>':0, '<unk>':1}
    vocab_ir_index.update(zip(vocab_ir, [item+2 for item in range(len(vocab_ir))]))

    
    vocab_ir_file_path = dir_path + args.vocab_ir_file
    ir_dic_str = json.dumps(vocab_ir_index)
    with open(vocab_ir_file_path, 'w') as vocab_ir_file:
        vocab_ir_file.write(ir_dic_str)


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value



def txt2json(args, ir_txt_file_path):
    mark_list = []
    start_index, end_index = [0, 0]
    ir_cnt = 1
    with open(ir_txt_file_path, 'r') as ir_txt_file:
        ir_lines = ir_txt_file.readlines()
        for i in range(0, len(ir_lines)):
            if ir_lines[i][0:6] == 'E:\\tmp' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
                ir_cnt += 1
        print('ir_cnt:\n', ir_cnt)
        mark_list.append([start_index, len(ir_lines)])

    dir_path = args.data_path + args.dataset
    vocab_ir_file_path = dir_path + args.vocab_ir_file
    vocab = json.loads(open(vocab_ir_file_path, 'r').readline())

    graph_dict = multidict()
    for i in range(0, ir_cnt):
        s_ind, e_ind = mark_list[i]
        #print("Graph Index: ", i)
        for j in range(s_ind+2, e_ind):
            edge_info_list = ir_lines[j].split()
            s_node_list = edge_info_list[0].split(':')
            e_node_list = edge_info_list[1].split(':')

            #print(edge_info_list)

            
            if (args.word_split_type == 'split'):
               
                s_node_index = int(s_node_list[0])
                if graph_dict[i][s_node_index]['wordid'] == {}:
                    if s_node_list[1] == 'control_label' or s_node_list[1] == 'return_point':
                        graph_dict[i][s_node_index]['wordid'] = [vocab.get(s_node_list[1], UNK_ID)]
                    else:
                        graph_dict[i][s_node_index]['wordid'] = []
                        s_node_word_list = s_node_list[1].split('_')
                        for k in range(0, len(s_node_word_list)):
                            graph_dict[i][s_node_index]['wordid'].append(vocab.get(s_node_word_list[k], UNK_ID))
                    #print("snode: %s index: %d" %(s_node_list[1], s_node_index))
                    #print(graph_dict[i][s_node_index]['wordid'])

                e_node_index = int(e_node_list[0])
                if graph_dict[i][e_node_index]['wordid'] == {}:
                    if e_node_list[1] == 'control_label' or e_node_list[1] == 'return_point':
                        graph_dict[i][e_node_index]['wordid'] = [vocab.get(e_node_list[1], UNK_ID)]
                    else:
                        graph_dict[i][e_node_index]['wordid'] = []
                        e_node_word_list = e_node_list[1].split('_')
                        for k in range(0, len(e_node_word_list)):
                            graph_dict[i][e_node_index]['wordid'].append(vocab.get(e_node_word_list[k], UNK_ID))
                    #print("enode: %s index: %d" %(e_node_list[1], e_node_index))
                    #print(graph_dict[i][e_node_index]['wordid'])
                    
            else:
                graph_dict[i][int(s_node_list[0])]['wordid'] = vocab.get(s_node_list[1], UNK_ID)
                graph_dict[i][int(e_node_list[0])]['wordid'] = vocab.get(e_node_list[1], UNK_ID)

            if graph_dict[i][int(s_node_list[0])]['snode'] == {}: 
                graph_dict[i][int(s_node_list[0])]['snode'] = [int(e_node_list[0])]
                #print('if \{None\}', graph_dict[i][int(s_node_list[0])]['node'])
            else: 
                graph_dict[i][int(s_node_list[0])]['snode'].append(int(e_node_list[0]))
                #print('multiple sons exist in line={}, i={}, s={}, node={}'.format(j, i, s_node_list[0], graph_dict[i][int(s_node_list[0])]['node']))

    graph_dict_str = json.dumps(graph_dict)
    ir_json_file_path = ir_txt_file_path[0:-3] + 'json'
    with open(ir_json_file_path, 'w') as ir_json_file:
        ir_json_file.write(graph_dict_str)    


def cnt_node_num(args):
    dir_path = args.data_path + args.dataset
    train_ir_file_path = dir_path + args.train_ir_file

    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()

    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_ir_file_path, 'r') as train_ir_file:
        ir_lines = train_ir_file.readlines()
        for i in range(0, len(ir_lines)): 
            line = ir_lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(ir_lines)]) 
    
    node_num = []
    for i in range(0, len(mark_list)):
        s_ind = mark_list[i][0]+1
        line = ir_lines[s_ind]
        n_num = int(line.split()[0])
        node_num.append(n_num)

    cnt = 0
    for i in range(0, len(node_num)):
        if node_num[i] > 1024:
            cnt += 1
    print('cnt = ', cnt)
        

def parse_args():
    parser = argparse.ArgumentParser("Prepare IR data for IREmbeder")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='github2/')

    parser.add_argument('--origin_ir_file', type=str, default='origin.ir.txt')
    parser.add_argument('--all_ir_file', type=str, default='all.ir.txt')

    parser.add_argument('--train_ir_file', type=str, default='train.ir.txt')
    parser.add_argument('--test_ir_file', type=str, default='test.ir.txt') 
    parser.add_argument('--train_ir_json_file', type=str, default='train.ir.json')
    parser.add_argument('--test_ir_json_file', type=str, default='test.ir.json')
   
    parser.add_argument('--vocab_ir_file', type=str, default='vocab.ir.json')

    parser.add_argument('--n_node', type=int, default=1024)
    parser.add_argument('--n_edge_types', type=int, default=1)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--annotation_dim', type=int, default=300)
    parser.add_argument('--ir_word_num', type=int, default=15000)

    parser.add_argument('--trainset_num', type=int, default=32000)
    parser.add_argument('--testset_num', type=int, default=1000)
    parser.add_argument('--testset_start_ind', type=int, default=32000)

    parser.add_argument('--word_split_type', type=str, default='split') # no_split

    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    #observe_data(args)
    preprocess_origin_ir(args)
    split_data(args)
    create_dict_file(args)
    
    
    dir_path = args.data_path + args.dataset
    ir_txt_train_file_path = dir_path + args.train_ir_file
    ir_txt_test_file_path = dir_path + args.test_ir_file
    txt2json(args, ir_txt_train_file_path)
    txt2json(args, ir_txt_test_file_path)
    
