import numpy as np
import configs
import argparse



def get_cfg_npy_info(lines, n_node, n_edge_types, state_dim, annotation_dim):
    
    all_num = 0 # count the number of cfgs
    for i in range(0, len(lines)):
        line = lines[i]
        if (line[0:10] == 'BeginFunc:'):
            all_num += 1
    #print('number of cfgs:\n', all_num)

    all_adjmat = np.zeros([all_num, n_node, n_node * n_edge_types * 2])
    #all_adjmat = np.zeros([all_num, n_node, n_node * n_edge_types * 2], dtype='float16')
    all_anno = np.zeros([all_num, n_node, annotation_dim])
    all_node_mask = np.zeros([all_num, n_node])
    cnt = 0
    for i in range(0, len(lines)):
        line = lines[i]

        if (line[0:10] == 'BeginFunc:'): 

            cfg_info_list = lines[i+1].split() # node_num and edge_num of current cfg
            node_num, edge_num = int(cfg_info_list[0]), int(cfg_info_list[1])

            save_node_feature_dict, save_edge_digit_list = {}, []
            for j in range(i+2, i+2+edge_num):
                start_node_info, edge_type, end_node_info = lines[j].split()
                start_node, start_node_feature = start_node_info.split(':')
                end_node, end_node_feature = end_node_info.split(':')   

                reset_edge_type = int(edge_type) 
                if reset_edge_type == 2:
                    reset_edge_type = 0

                
                if int(start_node) < n_node and int(end_node) < n_node:
                    save_edge_digit_list.append([int(start_node), reset_edge_type, int(end_node)])
                if int(start_node) < n_node:
                    save_node_feature_dict[int(start_node)] = int(start_node_feature)
                if int(end_node) < n_node:
                    save_node_feature_dict[int(end_node)] = int(end_node_feature)

            # adjmat: [n_node x (n_node * n_edge_types * 2)]
            adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
            # anno: [n_node x annotation_dim]
            anno = create_annotation_matrix(save_node_feature_dict, n_node, annotation_dim)
            # node_mask: [n_node]
            node_mask = [1 if k < node_num else 0 for k in range(n_node)]

            all_adjmat[cnt, :, :] = adjmat
            all_anno[cnt, :, :] = anno
            all_node_mask[cnt, :] = node_mask

            cnt += 1
            i += (edge_num + 1)
            '''
            if (cnt == 1):
                print('adjmat.size:\n', adjmat.shape)
                for i in range(0, len(adjmat)):
                    line = adjmat[i]
                    for j in range(0, len(line)):
                        if adjmat[i][j] == 1:
                            print('i:{}, j:{}'.format(i, j))
                print('anno.size:\n', anno.shape)
                for i in range(0, len(anno)):
                    line = anno[i]
                    for j in range(0, len(line)):
                        if anno[i][j] == 1:
                            print('i:{}, j:{}'.format(i, j))
                print('node_mask:\n', len(node_mask))
                for i in range(0, len(node_mask)):
                    if node_mask[i] == 1:
                        print(i)
            '''
    all_init_input = pad_anno(all_anno, n_node, state_dim, annotation_dim)
    # all_adjmat: [all_num x n_node x (n_node * n_edge_types * 2)]
    # all_init_input: [all_num x n_node x state_dim]
    # all_node_mask: [all_num x n_node]
    #print('type of adjmat:\n', type(adjmat))

    return all_adjmat, all_init_input, all_node_mask


def get_one_cfg_npy_info(lines, n_node, n_edge_types, state_dim):
    assert(lines[0][0:10] == 'BeginFunc:')
    
    cfg_info_list = lines[1].split() # node_num and edge_num of current cfg
    node_num, edge_num = int(cfg_info_list[0]), int(cfg_info_list[1])

    save_node_feature_dict, save_edge_digit_list = {}, []
    for j in range(2, 2+edge_num):
        start_node_info, end_node_info = lines[j].split()
        start_node, _ = start_node_info.split(':')
        end_node, _ = end_node_info.split(':')   

        
        if int(start_node) < n_node and int(end_node) < n_node:
            save_edge_digit_list.append([int(start_node), int(end_node)])

    # adjmat: [n_node x (n_node * n_edge_types * 2)]
    adjmat = create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types)
    # anno: [n_node x annotation_dim]
    init_input = np.zeros([n_node, state_dim])
    # node_mask: [n_node]
    node_mask = np.zeros([n_node])
    for k in range(node_num):
        node_mask[k] = 1
    #node_mask = [1 if k < node_num else 0 for k in range(n_node)]
    #print('type of adjmat:\n', type(adjmat))
    return init_input, adjmat, node_mask


def create_adjacency_matrix(save_edge_digit_list, n_node, n_edge_types):
    a = np.zeros([n_node, n_node * n_edge_types * 2])

    for edge in save_edge_digit_list:
        src_idx = edge[0]
        e_type = 0
        tgt_idx = edge[1]

        a[tgt_idx][(e_type) * n_node + src_idx] = 1
        a[src_idx][(e_type + n_edge_types) * n_node + tgt_idx] = 1

    return a
    

def create_annotation_matrix(save_node_feature_dict, n_node, annotation_dim):
    anno = np.zeros([n_node, annotation_dim])
    for node, node_feature in save_node_feature_dict.items():
        anno[node][node_feature] = 1

    return anno


def pad_anno(all_anno, n_node, state_dim, annotation_dim):
    padding = np.zeros((len(all_anno), n_node, state_dim - annotation_dim))
    all_init_input = np.concatenate((all_anno, padding), 2)
    return all_init_input

def pad_one_anno(anno, n_node, state_dim, annotation_dim):
    padding = np.zeros([n_node, (state_dim - annotation_dim)])
    init_input = np.concatenate((anno, padding), 1)
    return init_input


def split_data(args):
    index = np.load(args.shuffle_index_file)

    dir_path = args.data_path + args.dataset
    all_cfg_file_path = dir_path + args.all_cfg_file
    train_cfg_file_path = dir_path + args.train_cfg_file
    test_cfg_file_path = dir_path + args.test_cfg_file

    mark_list = []
    cnt = -1
    start_index = 0
    end_index = 0
    with open(all_cfg_file_path, 'r') as all_cfg_file:
        lines = all_cfg_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])
    print('all_num of cfg:\n', len(mark_list))

    with open(train_cfg_file_path, 'w') as train_cfg_file,  open(test_cfg_file_path, 'w') as test_cfg_file:
        for i in range(0, args.trainset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                train_cfg_file.write(lines[j])
        for i in range(args.testset_start_ind, args.testset_start_ind+args.testset_num):
            ind = index[i]
            for j in range(mark_list[ind][0], mark_list[ind][1]):
                test_cfg_file.write(lines[j])


def parse_args():
    parser = argparse.ArgumentParser("Prepare CFG data for CFGEmbeder")
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='github11/', help='name of dataset.c')
    
    parser.add_argument('--all_cfg_file', type=str, default='all.cfg.txt')
    parser.add_argument('--train_cfg_file', type=str, default='train.cfg.txt')
    parser.add_argument('--test_cfg_file', type=str, default='test.cfg.txt')  

    parser.add_argument('--n_node', type=int, default=150)
    parser.add_argument('--n_edge_types', type=int, default=1)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--annotation_dim', type=int, default=5)

    parser.add_argument('--trainset_num', type=int, default=39152)
    parser.add_argument('--testset_num', type=int, default=2000)
    parser.add_argument('--testset_start_ind', type=int, default=39152)

    parser.add_argument('--shuffle_index_file', type=str, default='shuffle_index.npy')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    split_data(args)
    
    '''
    dir_path = args.data_path + args.dataset
    train_cfg_file_path = dir_path + args.train_cfg_file
    mark_list = []
    start_index, end_index = [0, 0]
    with open(train_cfg_file_path, 'r') as train_cfg_file:
        lines = train_cfg_file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if line[0:10] == 'BeginFunc:' and i != 0:
                end_index = i
                mark_list.append([start_index, end_index])
                start_index = i
        mark_list.append([start_index, len(lines)])

    max_node_num = 0
    cnt = 0
    for i in range(0, len(mark_list)):
        line = lines[mark_list[i][0]+1]
        node_num = int(line.split()[0])
        
        #if node_num > max_node_num:
            #max_node_num = node_num
            #print(max_node_num)
        
        if node_num > 200:
            cnt += 1
    print(cnt)
    '''
