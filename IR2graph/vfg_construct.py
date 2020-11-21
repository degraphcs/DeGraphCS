import re
import networkx as nx
import logging
import os

error_dir = "E:\\tmp\\error\\"
runtime_error_file = error_dir+"runtime.txt"


logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(runtime_error_file)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(funcName)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


global_id = r'(?<!%")@["\w\d\.\-\_\$\\]+'
rgx_func_name = r'@[\"\w\d\._\$\\]+'
rgx_local_ident_no_perc = '[\"\@\d\w\.\-\_\:]+'
rgx_local_ident = '[%|@]' + rgx_local_ident_no_perc
rgx_global_ident = local_or_global_id = r'(' + global_id + r'|' + rgx_local_ident + r')'

base_type = r'(?:i\d+|double|float|opaque)\**'
immediate_value_int = r'(?<!\w)[-]?[0-9]+'
base_type_or_struct_name = ""


rgx_binary_operations = r"(add|fadd|sub|fsub|mul|fmul|udiv|sdiv|fdiv|urem|srem|frem|shl|lshr|ashr|and|or|xor)"
conversion_op = r"(trunc|zext|sext|fptrunc|fpext|fptoui|fptosi|uitofp|sitofp|ptrtoint|inttoptr|bitcast|addrspacecast)"
terminator_operations = ['ret','']
binary_operations = ['add','fadd','sub','fsub','mul','fmul','udiv','sdiv','fdiv','urem','srem','frem','shl','lshr','ashr','and','or','xor']
bitwise_binary_operations = ['shl','lshr','ashr','and','or','xor']
vector_operations = ['extractelement','insertelement','shufflevector']

cmp_tag = {"eq","ne","ugt","uge","ult","ule","sgt","sge","slt","sle"}
cmp_tag_simp = {
    "ugt":"gt","sgt":"gt","ogt":"gt",
    "uge":"ge","sge":"ge","oge":"ge",
    "ult":"lt","slt":"lt","olt":"lt",
    "ule":"le","sle":"le","ole":"le",
    "eq":"eq","seq":"eq","oeq":"eq","ueq":"eq",
    "ne":"ne","sne":"ne","une":"ne","one":"ne",
    "uno":"no"
}


opcode_simplify = {"fadd":"add","fsub":"sub","fmul":"mul",
                   "udiv": "div","sdiv": "div","fdiv": "div",
                   "urem": "rem","srem": "rem","frem": "rem"}
				   
node_replication = {}

label_tag = {}

alloca_addr = []

ano_addr = {}


control_tag = "control"
dataflow_tag = "dataflow"
repli_connector = "@_@"

def find_root(node):
    if node not in ano_addr:
        ano_addr[node] = node
        return  node
    elif node == ano_addr[node]:
        return node
    else:
        ano_addr[node] = find_root(ano_addr[node])
        return ano_addr[node]

def merge_node_pair(node1,node2):
    fa_1 = find_root(node1)
    fa_2 = find_root(node2)
    if fa_1 != fa_2:
        ano_addr[fa_1] = fa_2


def insert_node(G,node,label_id):
    if not G.has_node(node):
        
        G.add_node(node,label=label_id)

def insert_edge(G,pre,next,tag):
    if not G.has_edge(pre,next):
        G.add_edge(pre,next,label=tag)


def get_local_identifier(line):
    
    modif_line = re.sub(r'\"[^\s]*\"', '', line)
    m_loc = re.findall(rgx_local_ident, modif_line)
    
    if len(m_loc) > 0:
        to_remove = []
        for m in m_loc:
            if m + '*' in line:
                to_remove.append(m)
            if m[:2] == '%"':
                to_remove.append(m)
            if ' = phi ' + m in line:
                to_remove.append(m)
            if ' x ' + m in line:
                to_remove.append(m)
            if ' alloca ' + m in line and m+" = alloca " not in line:
                to_remove.append(m)
        if len(to_remove) > 0:
            m_loc = [m for m in m_loc if m not in to_remove]
    return m_loc


def get_label_identifier(line):
    
    m_label = m_label2 = list()
    if line.find('label') is not -1 or re.match(rgx_local_ident_no_perc + r':', line):
        m_label1 = re.findall('label (' + rgx_local_ident + ')', line)
        if re.match(r'; <label>:' + rgx_local_ident_no_perc + ':\s+', line):
            m_label2 = re.findall('<label>:(' + rgx_local_ident_no_perc + '):', line)
        elif 'invoke ' in line:
            m_label2 = re.findall('label (' + rgx_local_ident_no_perc + ')', line)
        else:
            m_label2 = re.findall('<label>:(' + rgx_local_ident_no_perc + ')', line) + \
                       re.findall(r'(' + rgx_local_ident_no_perc + r'):', line)
        for i in range(len(m_label2)):
           
            m_label2[i] = '%' + m_label2[i]
    m_label = m_label1 + m_label2
    return m_label


class Operation:
    def __init__(self,opcode,opnants):
        self.opcode = opcode
        self.opnants = opnants

    def get_params(self):
        params = [self.opcode]+self.opnants
        return params

def get_operation(line):
    loc = get_local_identifier(line)
    opcode = ""
    opnants = loc
    if re.match(rgx_local_ident + r' = ((tail )?(call|invoke) )', line):
      
        assignee = re.match(r'(' + rgx_local_ident + ') = ', line).group(1)
        func_name_ = re.search(r'(' + rgx_func_name + r')\(.*\)( to .*)?', line)
        if func_name_ is None:
            func_name_ = re.search(r'(' + rgx_local_ident + r')\(.*\)( to .*)?', line)
      
        if func_name_ is None:
            func_name_ = re.search(r'(' + rgx_local_ident + r') to .*', line)
        if func_name_ is None:
            opcode = "error"
        else:
            func_name = func_name_.group(1)
            if re.search(func_name + '(\(.*\))', line) is None:
                params = []
            else:
                param_line = re.search(func_name + '(\(.*\))', line).group(1)
                loc = get_local_identifier(param_line)
                if len(loc) > 0:
                    params = loc
                else:
                    params = []
            if re.match(rgx_local_ident + r' = ((tail )?call)', line):
                opcode = "call"
            else:
                opcode = "invoke"
            opnants = [func_name,assignee]+params
   
    elif re.match('((tail )?(call|invoke) )', line):
        func_name_ = re.search(r'(' + rgx_func_name + r')\(.*\)( to .*)?', line)
        if func_name_ is None:
            func_name_ = re.search(r'(' + rgx_local_ident + r')\(.*\)( to .*)?', line)
        # print(line)
        if func_name_ is None:
            func_name_ = re.search(r'(' + rgx_local_ident + r') to .*', line)
        if func_name_ is None:
            opcode = "error"
        else:
            func_name = func_name_.group(1)
            if re.search(func_name + '(\(.*\))', line) is None:
                params = []
            else:
                param_line = re.search(func_name + '(\(.*\))', line).group(1)
                loc = get_local_identifier(param_line)
                params = loc
            if re.match('((tail )?call)', line):
                opcode = "call"
            else:
                opcode = "invoke"
            opnants = [func_name,"void_res"]+params
    elif re.match("store ", line):
        opcode = "store"
        if len(loc) < 1:
            opcode = "error"
            logger.info(line)
            logger.info("store error")
        elif len(loc) == 1:
            addr = loc[0]
            if addr in node_replication:
                replic = addr+repli_connector+str(node_replication[addr])
                node_replication[addr] += 1
            else:
                replic = addr+repli_connector+"0"
                node_replication[addr] = 1
            loc.insert(0,replic)

    elif re.match(rgx_local_ident + r' = load', line):
        opcode = "load"
        if len(opnants)==1:
            opnants.append("const")

    elif re.match("ret ", line):
        if re.match("ret void", line):
            opcode = "ret_void"
        else:
            opcode = "ret"
            if len(loc) == 1:
                opnants = loc

    elif re.match("br ", line):
        opcode = "br"
        if len(opnants)==2:
            if "br i1 false," in line:
                opnants.insert(0,"false")
            elif "br i1 true," in line:
                opnants.insert(0,"true")
            else:
                logger.info(line)
                logger.info("br error")

    elif re.match("switch ", line):
        opcode = "switch"

    elif re.match(rgx_local_ident + r' = (atomicrmw )?' + rgx_binary_operations, line):
        if ' = atomicrmw ' in line:
            opcode = line.split(" ")[3]
        else:
            opcode = line.split(" ")[2]
        if opcode in opcode_simplify:
            opcode = opcode_simplify[opcode]
    elif re.match(rgx_local_ident + r' = fneg', line):
        opcode = "fneg"

    elif re.match(rgx_local_ident + r' = (alloca)', line):
        opcode = "alloca"
    elif re.match(rgx_local_ident + r' = getelementptr', line):
        opcode = "getelementptr"
    elif re.match(rgx_local_ident + r" = " + conversion_op + " .* to ", line):  
        opcode = "conversion"
    elif re.match(rgx_local_ident + r" = (icmp|fcmp) ", line):
        cmp_type = line.split(" ")[3]
        simp_cmp_type = cmp_tag_simp[cmp_type]
        opcode = "cmp_"+simp_cmp_type

    elif re.match(rgx_local_ident + r" = phi", line):
        loc = get_local_identifier(line)
        assignee = loc[0]
        phi_list = re.findall(r'\[ (%?' + rgx_local_ident_no_perc +
                        r'|true|false|<.*>|getelementptr inbounds \([ \d\w\[\]\*\.@,]+\)|.*' + global_id +
                        r'.*), (%?' + rgx_local_ident_no_perc + ') \],?', line)
        opcode = "phi"
        opnants = [assignee]
        for phi_ in phi_list:
            opnants += [phi_[0],phi_[1]]

    elif re.match(rgx_local_ident + r" = select", line):
        opcode = "select"
    
    else:
        logger.info("opcode miss:"+line)
    return Operation(opcode,opnants)


def test_get_operation(file_pos,line,operation):
    file = "other.txt"
    if re.match(rgx_local_ident + r' = ((tail )?(call|invoke) )', line):
        file = "call.txt"
    elif re.match('((tail )?(call|invoke) )', line):
        file = "call.txt"
    elif re.match("store ", line):
        file = "store.txt"
    elif re.match(rgx_local_ident + r' = load', line):
        file = "load.txt"
    elif re.match("ret ", line):
        file = "ret.txt"
    elif re.match("br ", line):
        file = "br.txt"
    elif re.match("switch ", line):
        file = "switch.txt"
    elif re.match(rgx_local_ident + r' = (atomicrmw )?' + rgx_binary_operations, line):
        file = "binary_op.txt"
    elif re.match(rgx_local_ident + r' = fneg', line):
        file = "binary_op.txt"
    elif re.match(rgx_local_ident + r' = (alloca)', line):
        file = "alloca.txt"
    elif re.match(rgx_local_ident + r' = getelementptr', line):
        file = "getelementptr.txt"
    elif re.match(rgx_local_ident + r" = " + conversion_op + " .* to ", line): 
        file = "conversion.txt"
    elif re.match(rgx_local_ident + r" = (icmp|fcmp) ", line):
        file = "cmp.txt"
    elif re.match(rgx_local_ident + r" = phi", line):
        file = "phi.txt"
    elif re.match(rgx_local_ident + r" = select", line):
        file = "select.txt"
    root = error_dir + file
    f_error = open(root,'a')
    f_error.write(file_pos+"\n")
    f_error.write(line+"\n")
    f_error.write(str(operation.get_params())+"\n")


def combine_operation(file_lines):
    separator = " "
    erase_token = "erase_token"
    for i in range(len(file_lines)):
        if re.match(r'switch', file_lines[i].strip()):
            for j in range(i + 1, len(file_lines)):
                if re.search(r'i\d+ -?\d+, label ' + rgx_local_ident, file_lines[j]):
                    
                    file_lines[i] += separator + file_lines[j] 
                    file_lines[j] = erase_token 
                else:
                    
                    file_lines[i] += ']' 
                    break
        elif re.search(r'invoke', file_lines[i]):
            if i + 1 < len(file_lines):
                if re.match(r'to label ' + rgx_local_ident + ' unwind label ' + rgx_local_ident, file_lines[i + 1]):
                    file_lines[i] += separator + file_lines[i + 1]  
                    file_lines[i + 1] = erase_token  
    new_file_lines = []
    for line in file_lines:
        line = line.strip()
        if line == "":
            continue
        elif line == erase_token:
            continue
        elif re.match("; Function Attrs: ",line):
            continue
        elif line == "]":
            continue
        elif line == "; No predecessors!":
            return False
            continue
        else:
            new_file_lines.append(line)
    pos = 0
    while(1):
        line = new_file_lines[pos]
        if re.match(r'define .* ' + rgx_func_name + '\(.*\)', line):
           break
        pos += 1
    return new_file_lines[pos:]

def build_cfg_graph(file):
    cfg = nx.DiGraph()
    file_lines = open(file).readlines()
   
    file_lines = combine_operation(file_lines)
    if not file_lines:
        return False,False
    index = 0
    prefix_label = "label_"
    label_name = prefix_label+"0"
    operation_list = []
    label_and_operations = {} 
    cfg.add_node(label_name,label=label_name)
    min_label_id = 10
    block_id = 1
    while (index < len(file_lines)):
        line = file_lines[index].strip()
        index += 1
        if re.match(r'define .* ' + rgx_func_name + '\(.*\)', line):
            func_name = re.match(r'define .* (' + rgx_func_name + ')\(.*\)', line).group(1)
            params = get_local_identifier(line)
            opcode = "func"
            operation = Operation(opcode,[func_name]+params)
            operation_list.append(operation)
            test_get_operation(file,line,operation)
        elif line.find("; <label>:") is not -1 or (line.find("; preds = ") is not -1):
            block_id += 1
            if len(operation_list) == 0:
                logger.info(file)
                logger.info("label name error")
            else:
                label_and_operations[label_name] = operation_list
                operation_list = []
            
            if line.find("; <label>:") is not -1:
                label_id = re.search('<label>:(' + rgx_local_ident_no_perc + ')', line).group(1)
                label_name = prefix_label+label_id
            else:
                label_id = line.split(":")[0]
                label_name = prefix_label+label_id
           
            cfg.add_node(label_name,label=label_name)
            loc = get_local_identifier(line)
            pre_labels = []
            for label_pre in loc:
                if label_pre[1:].isdigit():
                    min_label_id = min(int(min_label_id),int(label_pre[1:]))
                pre_label_name = prefix_label+label_pre[1:]
                pre_labels.append(pre_label_name)
                cfg.add_node(pre_label_name,label=pre_label_name)
                cfg.add_edge(pre_label_name,label_name)
        elif line.strip() == "}":  
            if len(operation_list) == 0:
                logger.info(file)
                logger.info("block empty error")
            else:
                
                label_and_operations[label_name] = operation_list
            break
        else:
            operation = get_operation(line)
            operation_list.append(operation)
    
    if min_label_id > 0 and block_id > 1:
        label_and_operations[prefix_label+str(min_label_id)] = label_and_operations['label_0']
        cfg.remove_node("label_0")
    return cfg,label_and_operations


def compress_cfg(cfg,label_and_operations):
    label_list = []
    for node in cfg.nodes():
        label_list.append(node)
    for label in label_list:
        pre_nodes = [pre for pre in cfg.predecessors(label)]
        if len(pre_nodes)==1:
            pre_node = pre_nodes[0]
            pre_next_nodes = [suc for suc in cfg.successors(pre_node)]
            if len(pre_next_nodes) == 1:
                
                last_operation = label_and_operations[pre_node][-1]
                if last_operation.opcode == "br" or last_operation.opcode == 'switch':
                    label_and_operations[pre_node].pop()
                label_and_operations[pre_node] += label_and_operations[label]
                next_nodes = [suc for suc in cfg.successors(label)]
                for next_node in next_nodes:
                    cfg.add_edge(pre_node,next_node)
                cfg.remove_node(label)
                label_list.remove(label)
                

    return cfg,label_and_operations,label_list


def find_last_definition(label,label_and_operations,current,address):
    operations = label_and_operations[label]
    last_definition = label
    last = current - 1
    while(last >= 0):
        operation = operations[last]
        opcode = operation.opcode
        
        if opcode == "store":
            opnants = operation.opnants
            if len(opnants) == 2:
                val = opnants[0] 
                addr = opnants[1]
            else:
                addr = "const"
                val = "empty"
                logger.info("store const exist")
            if address=="" or find_root(address)==find_root(addr):
                last_definition = val
                break
        last -= 1
    return last_definition


def find_last_load(label,label_and_operations,current,address):
    operations = label_and_operations[label]
    last_definition = label
    last = current - 1
    while(last >= 0):
        operation = operations[last]
        opcode = operation.opcode
       
        if opcode == "load":
            opnants = operation.opnants
            if len(opnants) == 2:
                addr_1 = opnants[0] 
                addr = opnants[1]
            else:
                addr_1 = "const"
                logger.info("load const exist")
            if address == addr_1:
                last_definition = addr_1
                break
        last -= 1
    return last_definition


def find_last_def_over_cfg(cfg,label,label_and_operations,current,address):
    searched_label = []
    label_array = [(label,current)]
    
    last_def_list = []
    while(len(label_array) > 0):
        searching_label,cur = label_array[0]
        label_array.remove((searching_label,cur))
        
        last_def = find_last_definition(searching_label,label_and_operations,cur,address)
        if last_def == searching_label:
            for pre_label in cfg.predecessors(searching_label):
                if pre_label not in searched_label:
                    searched_label.append(pre_label)
                    cur_pos = len(label_and_operations[pre_label])
                    label_array.append((pre_label,cur_pos))
        else:
            last_def_list.append(last_def)
    return last_def_list

def is_digit(str):
    str = str.replace("_","")
    str = str.replace("@","")
    return str.isdigit()

def get_accessible_node_list(G,begin_node):
    accessible_node_list = []
    nodes_to_be_checked = []
    have_been_checked = []
    nodes_to_be_checked.append(begin_node)
    while(len(nodes_to_be_checked) > 0):
        node = nodes_to_be_checked.pop()
        if not(node[0]=="%" and node[1:].isdigit()) and node != begin_node:
            accessible_node_list.append(node)
            continue
        have_been_checked.append(node)
        for neibour in G.neighbors(node):
            if neibour not in have_been_checked:
                nodes_to_be_checked.append(neibour)
    return accessible_node_list


def remove_redunt_node(G):
    new_graph = nx.DiGraph()
    
    for node in G.nodes():
        if node[0]=="%" and node[1:].isdigit():
            continue
        if not new_graph.has_node(node):
            label_id = G.nodes[node]['label']
            new_graph.add_node(node,label=label_id)
        neibours = get_accessible_node_list(G,node)
        for neibour in neibours:
            if not new_graph.has_node(neibour):
                label_id = G.nodes[neibour]['label']
                new_graph.add_node(neibour,label=label_id)
            if "label_" in node or "label_" in neibour:
                tag = "control"
            elif G.has_edge(node,neibour) and G[node][neibour]['label'] =="control":
                tag = "control"
            else:
                tag = "dataflow"
            if not new_graph.has_edge(node,neibour):
                insert_edge(new_graph, node, neibour, tag)
    node_list = []
    for node in new_graph.nodes():
        if node not in node_list:
            node_list.append(node)
    for node in node_list:
        if new_graph.in_degree(node)==0 and new_graph.out_degree(node)==0:
            new_graph.remove_node(node)
    return new_graph

def get_no_term_node_list(G,begin_node):
    accessible_node_list = []
    nodes_to_be_checked = []
    have_been_checked = []
    nodes_to_be_checked.append(begin_node)
    while(len(nodes_to_be_checked) > 0):
        node = nodes_to_be_checked.pop()
        if not(node[0]=="%" and is_digit(node[1:])) and node != begin_node:
            accessible_node_list.append(node)
            continue
        have_been_checked.append(node)
        for neibour in G.neighbors(node):
            if neibour not in have_been_checked:
                nodes_to_be_checked.append(neibour)
    return accessible_node_list

def remove_termidiate(G):
    new_graph = nx.DiGraph()
    for node in G.nodes():
        if node[0] == "%" and is_digit(node[1:]):  
            continue
        if not new_graph.has_node(node):
            label_id = G.nodes[node]['label']
            new_graph.add_node(node, label=label_id)
        neibours = get_no_term_node_list(G, node)
        for neibour in neibours:
            if not new_graph.has_node(neibour):
                label_id = G.nodes[neibour]['label']
                new_graph.add_node(neibour, label=label_id)
            if "label_" in node or "label_" in neibour:
                tag = "control"
            elif G.has_edge(node, neibour) and G[node][neibour]['label'] == "control":
                tag = "control"
            else:
                tag = "dataflow"
            if not new_graph.has_edge(node, neibour):
                insert_edge(new_graph,node, neibour, tag)
    node_list = []
    for node in new_graph.nodes():
        if node not in node_list:
            node_list.append(node)
    for node in node_list:
        if new_graph.in_degree(node) == 0 and new_graph.out_degree(node) == 0:
            new_graph.remove_node(node)
    return new_graph

def rename_node(G,origin,new_name):
    pre_nodes = [pre for pre in G.predecessors(origin)]
    next_nodes = [suc for suc in G.successors(origin)]
    G.add_node(new_name,label=G.nodes[origin]['label'])
    for pre_node in pre_nodes:
        insert_edge(G,pre_node,new_name,G[pre_node][origin]['label'])
    for next_node in next_nodes:
        insert_edge(G,new_name,next_node,G[origin][next_node]['label'])
    if origin in variable_type:
        variable_type[new_name] = variable_type[origin]
    G.remove_node(origin)

def remove_postfix(str):
    sub_list = str.split("_")
    res = sub_list[0]
    if len(sub_list)>1:
        for ind in range(1,len(sub_list)-1):
            res += "_"+sub_list[ind]
    return res

def graph_optimazition(vfg,store_operations,load_operations):
    
    node_name = {}
    while(1):
        ischanged = False
        for label,operation in store_operations:
            val = operation.opnants[0]
            addr = operation.opnants[1]
            if addr[1:].isdigit() and addr not in node_name:
                if not val[1:].isdigit():
                    val_name = val
                elif val in node_name:
                    val_name = node_name[val]
                else:
                    continue
                if repli_connector in val_name:
                    val_real_name = val_name.split(repli_connector)[0]
                else:
                    val_real_name = val_name
                if val_real_name in node_replication:
                    addr_name = val_real_name + repli_connector + str(node_replication[val_real_name])
                    node_replication[val_real_name] += 1
                else:
                    addr_name = val_real_name + repli_connector +"0"
                    node_replication[val_real_name] = 1
                node_name[addr] = addr_name
                ischanged = True
        for label,operation in store_operations:
            val = operation.opnants[0]
            addr = operation.opnants[1]
            if val[1:].isdigit() and val not in node_name:
                if not addr[1:].isdigit():
                    addr_name = addr
                elif addr in node_name:
                    addr_name = node_name[addr]
                else:
                    continue
                if repli_connector in addr_name:
                    addr_real_name = addr_name.split(repli_connector)[0]
                else:
                    addr_real_name = addr_name
                if addr_real_name in node_replication:
                    val_name = addr_real_name + repli_connector + str(node_replication[addr_real_name])  
                    node_replication[addr_real_name] += 1
                else:
                    val_name = addr_real_name + repli_connector +"0"
                    node_replication[addr_real_name] = 1
                node_name[val] = val_name
                ischanged = True
        if not ischanged:
            break
    load_node_name = {}
    while(1):
        ischanged = False
        for label,operation in load_operations:
            val = operation.opnants[0]
            addr = operation.opnants[1]
            if val[1:].isdigit() and val not in node_name and val not in load_node_name: 
                if not addr[1:].isdigit():
                    addr_name = addr
                elif addr in node_name:
                    addr_name = node_name[addr]
                elif addr in load_node_name:
                    addr_name = load_node_name[addr]
                else:  
                    continue
                if repli_connector in addr_name:
                    addr_real_name = addr_name.split(repli_connector)[0]
                else:
                    addr_real_name = addr_name
                if addr_real_name in node_replication:
                    val_name = addr_real_name + repli_connector + str(node_replication[addr_real_name])  
                    node_replication[addr_real_name] += 1
                else:
                    val_name = addr_real_name + repli_connector + "0"
                    node_replication[addr_real_name] = 1
                load_node_name[val] = val_name
                ischanged = True
        if not ischanged:
            break
    while (1):
        ischanged = False
        for label, operation in store_operations:
            val = operation.opnants[0]
            addr = operation.opnants[1]
            if addr[1:].isdigit() and addr not in node_name:  
                if addr in load_node_name:
                    node_name[addr] = load_node_name[addr]
                if not val[1:].isdigit():
                    val_name = val
                elif val in node_name:
                    val_name = node_name[val]
                elif val in load_node_name:
                    val_name = load_node_name[val]
                else:  
                    continue
               
                if repli_connector in val_name:
                    val_real_name = val_name.split(repli_connector)[0]
                else:
                    val_real_name = val_name
                if val_real_name in node_replication:
                    addr_name = val_real_name + repli_connector + str(node_replication[val_real_name]) 
                    node_replication[val_real_name] += 1
                else:
                    addr_name = val_real_name + repli_connector + "0"
                    node_replication[val_real_name] = 1
                node_name[addr] = addr_name
                ischanged = True
        
        for label, operation in store_operations:
            val = operation.opnants[0]
            addr = operation.opnants[1]
            if val[1:].isdigit() and val not in node_name:  
                if val in load_node_name:
                    node_name[val] = load_node_name[val]
                if not addr[1:].isdigit():
                    addr_name = addr
                elif addr in node_name:
                    addr_name = node_name[addr]
                elif addr in load_node_name:
                    addr_name = load_node_name[addr]
                else:  
                    continue
            
                if repli_connector in addr_name:
                    addr_real_name = addr_name.split(repli_connector)[0]
                else:
                    addr_real_name = addr_name
                if addr_real_name in node_replication:
                    val_name = addr_real_name + repli_connector + str(node_replication[addr_real_name])  
                    node_replication[addr_real_name] += 1
                else:
                    val_name = addr_real_name + repli_connector + "0"
                    node_replication[addr_real_name] = 1
                node_name[val] = val_name
                ischanged = True
        if not ischanged:  
            break
    
    for node in node_name:
        rename_node(vfg,node,node_name[node])
    f_w = open("E:\\tmp\\error\\vfg_o0.txt", 'w')

    f_w.close()
    
    for label,operation in store_operations:
        
        addr = operation.opnants[1]
        if addr in node_name:
            real_addr = node_name[addr]
        else:
            real_addr = addr
        if vfg.out_degree(real_addr) == 0:
            vfg.remove_node(real_addr)
    f_w = open("E:\\tmp\\error\\vfg_o1.txt", 'w')
    
    f_w.close()
   

    for label,operation in store_operations:
        val = operation.opnants[0]
        addr = operation.opnants[1]
        if addr in node_name:
            real_addr = node_name[addr]
        else:
            real_addr = addr
        if val in node_name:
            real_val = node_name[val]
        else:
            real_val = val
        if vfg.has_edge(real_val,real_addr):
            vfg.remove_edge(real_val,real_addr)
    
    new_vfg = remove_redunt_node(vfg)
    
    new_new_vfg = remove_termidiate(new_vfg)
    return new_new_vfg

def output_content(content):
    if content in variable_type:
        content_type = variable_type[content]
    else:
        content_type = ""
    content = content.replace(repli_connector,"_")
    content = content.replace("@","")
    if "#" in content:
        content = content.split("#")[-1]
    if "label_" in content:
        if content in label_tag:
            clean_content = label_tag[content]
        else:
            clean_content = "control_label"
    elif is_digit(content[1:]):
        clean_content = "intermediate"
    else:
        clean_content = content.replace("%","")
    if content_type:
        clean_content = content_type+"_"+clean_content
    return clean_content


def split_word_by_underline(word):
    new_word = ""
    for index in range(len(word)):
        if word[index].isupper():
            if index == 0 or (index+1<len(word) and word[index+1].isupper())\
                    or (index-1>=0 and word[index-1].isupper()):
                if index+1<len(word) and not word[index+1].isupper() and (index-1>=0 and word[index-1].isupper()):
                    new_word += word[index].lower() + "_"
                else:
                    new_word += word[index].lower()
            else:
                new_word += "_"+word[index].lower()
        else:
            new_word += word[index]
    return new_word


def print_graph(G,f_result,f_w):
    
    f_result.write(str(G.number_of_nodes()) + " " + str(G.number_of_edges()) + "\n")
    
    index = []
   
    for node in G.nodes():
        
        if node not in index:
            index.append(node)
        for neibour in G.neighbors(node):
            if neibour not in index:
                index.append(neibour)
            output_pre_id = index.index(node)
            output_next_id = index.index(neibour)
            if G[node][neibour]['label'] == "control":
                tag = 1
            else:
                tag = 0
           
            f_result.write(str(output_pre_id) + ":" + split_word_by_underline(output_content(node)) + " " +
                     str(output_next_id) + ":" + split_word_by_underline(output_content(neibour)) +" "+str(tag)+ "\n")
            
            if G.in_degree(node) == 0:
                if (output_pre_id,node) not in attension_list:
                    attension_list.append((output_pre_id,node))
            if G.out_degree(neibour) == 0:
                if (output_next_id,neibour) not in attension_list:
                    attension_list.append((output_next_id,neibour))
    for id,node in attension_list:
        f_w.write(str(id)+":"+split_word_by_underline(output_content(node))+"\n")
    
def get_type_identifier(line,opnants):

    modif_line = re.sub(r'\"[^\s]*\"', '', line)
    m_loc = re.findall(rgx_local_ident, modif_line)
    for opnant in opnants:
        if opnant in m_loc:
            m_loc.remove(opnant)

    if len(m_loc) > 0:
        to_remove = []
        for m in m_loc:
            if m[:2] == '%"':
                to_remove.append(m)
            if ' alloca ' + m in line and m+" = alloca " not in line:
                to_remove.append(m)
        if len(to_remove) > 0:
            m_loc = [m for m in m_loc if m not in to_remove]
    return m_loc


def clean_type(type):
    type = type.replace("@","")
    type = type.replace("%", "")
    type = type.replace("struct.", "")
    type = type.replace("union","")
    return type


def get_local_type(line,opnants):
    words = line.split(",")
    res = {}
    for word in words:
        
        idens = get_local_identifier(word)
        opnant = ""
        for iden in idens:
            if iden in opnants:
                opnant = iden
        
        if opnant:
            type = get_type_identifier(word,opnants)
            if type and len(type) == 1:
                res[opnant] = split_word_by_underline(clean_type(type[0]))
    return res


def get_ir_type(root):
    v_type = {}
    f = open(root)
    contents = f.readlines()
    f.close()
    for content in contents:
        opnants = get_local_identifier(content)
        
        type = get_local_type(content, opnants)
        
        if type:
            for var in type:
                if var not in v_type:
                    v_type[var] = type[var]
    return v_type

def build_control_flow_and_data_flow(cfg,label_list,label_and_operations):
    vfg = nx.DiGraph()
    unique_id = 0
    prefix_label = "label_"
    load_operation_list = []
    store_operation_list = []
    label_control_list = []
    def_list = {}
    call_and_cmp_list = {}
    function_name = ""
    first_label = label_list[0]
    
    for label in label_list:
        def_list[label] = []
        vfg.add_node(label,label=label)
        operations = label_and_operations[label]
        for operation in operations:
            opcode = operation.opcode
            opnants = operation.opnants
            if opcode == "call" or opcode == "invoke":
                
                funcname = opnants[0]
                return_val = opnants[1]
                func_node = str(unique_id)+"#"+funcname
                unique_id += 1
                
                if label not in call_and_cmp_list:
                    call_and_cmp_list[label] = [func_node]
                else:
                    call_and_cmp_list[label].append(func_node)
                if len(opnants)==2:
                    if return_val == "void_res":
                        insert_node(vfg,func_node,label)
                    else:
                        insert_node(vfg,func_node,label)
                        insert_node(vfg, return_val,label)
                        insert_edge(vfg,func_node,return_val,dataflow_tag)
                else:
                    insert_node(vfg, func_node, label)
                    if return_val != "void_res":
                        insert_node(vfg,return_val, label)
                        insert_edge(vfg,func_node, return_val,dataflow_tag)
                    params = opnants[2:]
                    for param in params:
                        insert_node(vfg,param,label)
                        insert_edge(vfg,param,func_node,dataflow_tag)
            elif opcode == "func":
                
                funcname = opnants[0]
                insert_node(vfg,funcname,label)
                function_name = funcname
                if len(opnants) > 1:
                    for id in range(1,len(opnants)):
                        param = opnants[id]
                        insert_node(vfg,param,label)
                        insert_edge(vfg,funcname,param,dataflow_tag)
            elif opcode =="ret_void":
                op_node = str(unique_id) + "#ret_void"
                unique_id += 1
                vfg.add_node(op_node,label=label)
                
                current = operations.index(operation)
                last_definition = find_last_definition(label,label_and_operations,current,"")
                insert_node(vfg,last_definition,label)
                insert_edge(vfg,last_definition,op_node,control_tag)
            elif opcode=="store":
                store_operation_list.append((label,operation))
                
                val = opnants[0]
                addr = opnants[1]
                insert_node(vfg,val,label)
                insert_node(vfg, addr, label)
                insert_edge(vfg, val,addr,dataflow_tag)
                
                val_label = vfg.nodes[val]['label']
                if val_label in def_list:
                    if val not in def_list[val_label]:
                        def_list[val_label].append(val)
                else:
                    def_list[val_label] = [val]
            elif opcode=="load":
                val = opnants[0]
                addr = opnants[1]
                insert_node(vfg,val,label)
                insert_node(vfg,addr,label)
                merge_node_pair(val,addr)
                load_operation_list.append((label,operation))
            elif opcode=="ret":
                op_node = str(unique_id) + "#ret"
                unique_id += 1
                insert_node(vfg,op_node,label)
                if len(opnants) != 0:
                    
                    ret_val = opnants[0]
                    insert_node(vfg, ret_val, label)
                    insert_edge(vfg, ret_val, op_node,dataflow_tag)
                
                current = operations.index(operation)
                last_definition = find_last_definition(label, label_and_operations, current, "")
                insert_node(vfg,last_definition,label)
                insert_edge(vfg, last_definition, op_node, control_tag)  
                if op_node not in def_list:
                    def_list[label] = [op_node]
                else:
                    def_list[label].append(op_node)
            elif opcode=="br":
                
                if len(opnants)==3:
                    
                    val = opnants[0]
                    label_true = prefix_label+opnants[1][1:]
                    label_false = prefix_label + opnants[2][1:]
                    insert_node(vfg,val,label)
                    insert_node(vfg, label_true, label_true)
                    insert_node(vfg, label_false, label_false)
                    
                    insert_edge(vfg, val, label_true, control_tag)
                    insert_edge(vfg, val, label_false, control_tag)
                    
                    if label_true in label_tag:
                        logger.info("label tag confused")
                    else:
                        label_tag[label_true] = "label_true"
                    if label_false in label_tag:
                        logger.info("label tag confused")
                    else:
                        label_tag[label_false] = "label_false"
                    
                    label_control_list.append(label_true)
                    label_control_list.append(label_false)
                elif len(opnants)==1:
                    pass
            elif opcode=="switch":
                val = opnants[0]
                insert_node(vfg,val,label)
                label_list = opnants[1:]
                
                for label_id in label_list:
                    switch_label = prefix_label+label_id[1:]
                    insert_node(vfg,switch_label,switch_label)
                    insert_edge(vfg, val,switch_label,control_tag)
                    
                    label_control_list.append(switch_label)
            elif opcode in binary_operations:
                op_node = str(unique_id) + "#"+opcode  
                unique_id += 1
                insert_node(vfg, op_node, label)
                res = opnants[0]
                insert_node(vfg,res,label)
                insert_edge(vfg, op_node,res,dataflow_tag)
                for val in opnants[1:]:
                    insert_node(vfg,val,label)
                    insert_edge(vfg, val,op_node,dataflow_tag)
            elif opcode == "fneg":
                op_node = str(unique_id) + "#" + opcode  
                unique_id += 1
                insert_node(vfg, op_node, label)
                res = opnants[0]
                insert_node(vfg, res, label)
                insert_edge(vfg, op_node, res, dataflow_tag)
                for val in opnants[1:]:
                    insert_node(vfg, val, label)
                    insert_edge(vfg, val, op_node, dataflow_tag)
            elif opcode == "alloca":
                if len(opnants) > 0:
                    alloca_addr.append(opnants[0])
            elif opcode == "getelementptr":
                op_node = str(unique_id) + "#" + opcode  
                unique_id += 1
                insert_node(vfg, op_node, label)
                res = opnants[0]
                insert_node(vfg,res,label)
                insert_edge(vfg, op_node,res,dataflow_tag)
                for val in opnants[1:]:
                    insert_node(vfg,val,label)
                    insert_edge(vfg, val,op_node,dataflow_tag)
            elif opcode == "conversion":  
                if len(opnants) == 1:
                    continue
                else:
                    res = opnants[0]
                    val = opnants[1]
                    insert_node(vfg,res,label)
                    insert_node(vfg,val,label)
                    insert_edge(vfg, val,res,dataflow_tag)
            elif "cmp_" in opcode:
                op_node = str(unique_id) + "#" + opcode  
                unique_id += 1
                insert_node(vfg, op_node, label)
                res = opnants[0]
                insert_node(vfg, res, label)
                insert_edge(vfg,op_node, res, dataflow_tag)

                if label not in call_and_cmp_list:
                    call_and_cmp_list[label] = [op_node]
                else:
                    call_and_cmp_list[label].append(op_node)
                for val in opnants[1:]:
                    insert_node(vfg, val, label)
                    insert_edge(vfg,val, op_node, dataflow_tag)
            elif opcode == "phi":
                
                res = opnants[0]
                insert_node(vfg, res, label)
                ind = 1
                while(ind < len(opnants)):
                    val = opnants[ind]
                    src_label = opnants[ind+1]
                    if "%" == val[0] or "@"==val[0]:
                        insert_node(vfg,val,label)
                        insert_edge(vfg,val,res,dataflow_tag)
                    else:
                        if 'true'==val or 'false'==val:
                            insert_node(vfg,val,label)
                            insert_edge(vfg,val,res,dataflow_tag)
                        else:
                            logger.info("const phi\n")
                    ind += 2
            elif opcode == "select":
                op_node = str(unique_id) + "#" + opcode  
                unique_id += 1
                insert_node(vfg, op_node, label)
                
                res = opnants[0]
                insert_node(vfg, res, label)
                insert_edge(vfg,op_node, res, dataflow_tag)
                for val in opnants[1:]:
                    insert_node(vfg, val, label)
                    insert_edge(vfg, val, op_node, dataflow_tag)
            elif opcode == "error":
                continue
            else:
                logger.info("opcode miss????")
    
    
    fw = open("E:\\tmp\\error\\dataflow_0.txt", 'w')
    
    fw.close()
    
    if first_label not in label_control_list:
        label_control_list.append(first_label)
    for label in label_control_list:
        
        definitions = def_list[label]
        for definition in definitions:
            
            has_father = False
            checked = [definition]
            check_array = [definition]
            while (len(check_array) > 0):
                checking = check_array[0]
                check_array.remove(checking)
                if definition != checking and checking in definitions:
                    
                    has_father = True
                    break
                else:
                    for pre_node in vfg.predecessors(checking):
                       
                        if pre_node not in checked and vfg.nodes[pre_node]['label'] == label:  
                            if vfg[pre_node][checking]['label'] == 'control':  
                                continue
                            checked.append(pre_node)
                            check_array.append(pre_node)
            if not has_father:
                if label == first_label:
                    insert_edge(vfg, function_name, definition, control_tag)
                else:
                    insert_edge(vfg, label, definition, control_tag)
        if label not in call_and_cmp_list:
            continue
        for op_node in call_and_cmp_list[label]:
            
            if label == first_label:
                insert_edge(vfg, function_name, op_node, control_tag)
            else:
                insert_edge(vfg, label, op_node, control_tag)
    
    fw = open("E:\\tmp\\error\\control_flow_0.txt", 'w')
    
    fw.close()
    
    for label,operation in load_operation_list:
        opnants = operation.opnants
        if len(opnants) != 2:
            logger.info("load op error\n")
        else:
            res = opnants[0]
            addr = opnants[1]
            
            current = label_and_operations[label].index(operation)
            last_load = find_last_load(label,label_and_operations,current,addr)
            insert_node(vfg,res,label)
            origin_vals = find_last_def_over_cfg(cfg, label, label_and_operations, current, addr)
            if res in origin_vals:
                origin_vals.remove(res)
            
            if len(origin_vals):
                for origin_val in origin_vals:
                    
                    insert_edge(vfg, origin_val,res,dataflow_tag)
            elif last_load != label:
                insert_node(vfg,last_load,label)
                insert_edge(vfg, last_load,res,dataflow_tag)
            else:
                insert_edge(vfg, addr, res, dataflow_tag)
    
    fw = open("E:\\tmp\\error\\dataflow_1.txt",'w')
    
    fw.close()
    
    for label, operation in store_operation_list:  
        current = label_and_operations[label].index(operation)
        opnants = operation.opnants
        if len(opnants) == 2:
            val = opnants[0]
            addr = opnants[1]
        else:
            logger.info("load over 3 or less 2?")
        origin_vals = find_last_def_over_cfg(cfg, label, label_and_operations, current, addr)
        if val in origin_vals:
            origin_vals.remove(val)
        
        if len(origin_vals) == 0:
            
            pass
        else:
            for origin_val in origin_vals:
                insert_edge(vfg, origin_val, val, control_tag)
    
    fw = open("E:\\tmp\\error\\control_flow_1.txt", 'w')
    
    fw.close()
    remve =[]
    for node in vfg.nodes():
        if vfg.in_degree(node)==0 and vfg.out_degree(node)==0:
            remve.append(node)
            
    for a in remve:
        vfg.remove_node(a)
    return vfg
    
    vfg = graph_optimazition(vfg,store_operation_list,load_operation_list)
    
    fw = open("E:\\tmp\\error\\vfg_final.txt", 'w')
    
    fw.close()
    return vfg

root = "E:\\tmp\\origin_datasets\\"
ir_list_file = r"E:\tmp\datasets\8-18\datasets\addr.txt"
f_w = open(r"E:\tmp\datasets\8-18\datasets\ir_no_opt.txt",'w')
fw_1 = open("E:\\tmp\\database\\ir_database_0_extra.txt",'w')
node_num = 0
min_value = 100
no_file = 0
variable_type = {}
runtag =False
with open(ir_list_file) as f:
    for line in f:
        line = line.strip()
        line = root+line+"#IR.txt"
        print(line)
        if not os.path.exists(line):
            print("no")
            continue
        node_replication = {}
        label_tag = {}
        alloca_addr = []
        ano_addr = {}
        attension_list = []
        variable_type = get_ir_type(line)
        cfg,label_and_operations = build_cfg_graph(line)
        if cfg == False:
            continue
        cfg,label_and_operations,label_list = compress_cfg(cfg,label_and_operations)
        
        vfg = build_control_flow_and_data_flow(cfg,label_list,label_and_operations)
        if vfg.number_of_nodes() > 1500 or vfg.number_of_nodes()<3:
            continue
        fw_1.write(line+"\n")
        f_w.write(line + "\n")
        node_num = max(node_num,vfg.number_of_nodes())
        print_graph(vfg, f_w, fw_1)
        
print(no_file)
print(node_num)
