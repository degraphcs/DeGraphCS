import javalang
import json
from tqdm import tqdm
import collections
import sys
import re

def process_source(file_name, save_file):
    with open(file_name, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(save_file, 'w+', encoding='utf-8') as save:
        for line in lines:
            code = line.strip()
            tokens = list(javalang.tokenizer.tokenize(code))
            tks = []
            for tk in tokens:
                if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                    tks.append('STR_')
                elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                    tks.append('NUM_')
                elif tk.__class__.__name__ == 'Boolean':
                    tks.append('BOOL_')
                else:
                    tks.append(tk.value)
            save.write(" ".join(tks) + '\n')


def get_ast(file_name, w):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(w, 'w+', encoding='utf-8') as wf:
        ign_cnt = 0
        for line in tqdm(lines):
            code = line.strip()
            print('code = ', code)
            tokens = javalang.tokenizer.tokenize(code)
            token_list = list(javalang.tokenizer.tokenize(code))
            length = len(token_list)
            #print('tokens = ', token_list)
            parser = javalang.parser.Parser(tokens)
            try:
                tree = parser.parse_member_declaration()
            except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                print(code)
                continue
            flatten = []
            for path, node in tree:
                flatten.append({'path': path, 'node': node})

            ign = False
            outputs = []
            stop = False
            for i, Node in enumerate(flatten):
                d = collections.OrderedDict()
                path = Node['path']
                node = Node['node']
                children = []
                for child in node.children:
                    child_path = None
                    if isinstance(child, javalang.ast.Node):
                        child_path = path + tuple((node,))
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                                children.append(j)
                    if isinstance(child, list) and child:
                        child_path = path + (node, child)
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path']:
                                children.append(j)
                d["id"] = i
                d["type"] = str(node)
                if children:
                    d["children"] = children
                value = None
                if hasattr(node, 'name'):
                    value = node.name
                elif hasattr(node, 'value'):
                    value = node.value
                elif hasattr(node, 'position') and node.position:
                    for i, token in enumerate(token_list):
                        if node.position == token.position:
                            pos = i + 1
                            value = str(token.value)
                            while (pos < length and token_list[pos].value == '.'):
                                value = value + '.' + token_list[pos + 1].value
                                pos += 2
                            break
                elif type(node) is javalang.tree.This \
                        or type(node) is javalang.tree.ExplicitConstructorInvocation:
                    value = 'this'
                elif type(node) is javalang.tree.BreakStatement:
                    value = 'break'
                elif type(node) is javalang.tree.ContinueStatement:
                    value = 'continue'
                elif type(node) is javalang.tree.TypeArgument:
                    value = str(node.pattern_type)
                elif type(node) is javalang.tree.SuperMethodInvocation \
                        or type(node) is javalang.tree.SuperMemberReference:
                    value = 'super.' + str(node.member)
                elif type(node) is javalang.tree.Statement \
                        or type(node) is javalang.tree.BlockStatement \
                        or type(node) is javalang.tree.ForControl \
                        or type(node) is javalang.tree.ArrayInitializer \
                        or type(node) is javalang.tree.SwitchStatementCase:
                    value = 'None'
                elif type(node) is javalang.tree.VoidClassReference:
                    value = 'void.class'
                elif type(node) is javalang.tree.SuperConstructorInvocation:
                    value = 'super'

                if value is not None and type(value) is type('str'):
                    d['value'] = value
                if not children and not value:
                    # print('Leaf has no value!')
                    print(type(node))
                    print(code)
                    ign = True
                    ign_cnt += 1
                    # break
                outputs.append(d)
            if not ign:
                wf.write(json.dumps(outputs))
                wf.write('\n')
    print(ign_cnt)

code_merge_function = ""
def dfs(num):
   
    if(json_ast_data[num].get('value') != None):
       idx = json_ast_data[num]['type'].find('(')
       dict_node[json_ast_data[num]['type'][0:idx]+str(json_ast_data[num]['id'])] = json_ast_data[num]['value']
       if(json_ast_data[num]['type'].find('FormalParameter') != -1 and num != 0):
           child_num = json_ast_data[num]['children'][0]
           dict_variable[json_ast_data[num]['value']] = json_ast_data[child_num]['value']
       
    if(json_ast_data[num].get('children') != None):
      for i in json_ast_data[num]['children']:
        dfs(i)
source_code = ""
def dfs1(num):
    global source_code
    if(json_ast_data[num]['type'][0:20].find('ClassCreator') != -1 and num != 0):
        child_nodes = json_ast_data[num]['children']
        class_name = json_ast_data[child_nodes[0]]['value']
        method_name_parameters = []
        method_name_parameters_type = []
        for j in range(1, len(child_nodes)):
            method_name_parameter = json_ast_data[child_nodes[j]]['value']
            method_name_parameters.append(method_name_parameter)
            if(dict_variable.get(method_name_parameter) != None):
                   method_name_parameters_type.append(dict_variable[method_name_parameter])
            else:
                   method_name_parameter_type = method_name_parameter[0].swapcase()+method_name_parameter[1:]
                   method_name_parameters_type.append(method_name_parameter_type)
        print('in new class_name = ', class_name, ' method_name_parameters = ', method_name_parameters)
        try:
             class_type = class_name
             str_insert_class = 'class '+ class_type +'{'
             index_insert_class = source_code.find(str_insert_class) + 7 + len(class_type)
             print('index_insert_class  = ', index_insert_class)
             pre_string = source_code[0:index_insert_class]
             nxt_string = source_code[index_insert_class:]
             pre_string = pre_string + '\n'+ "public " +  + " " + class_type + "("
             for idd in range(len(method_name_parameters_type)):
                 pre_string = pre_string + method_name_parameters_type[idd] + " " + method_name_parameters[idd] + ","
             if(len(method_name_parameters_type) > 0):
                 pre_string = pre_string[:-1]
             pre_string = pre_string + "){}"
             source_code = pre_string + nxt_string
        except:
               print("Wrong XXXX : ", class_type, " not in the dict")
    if(json_ast_data[num]['type'][0:20].find('MethodInvocation') != -1 and num != 0): 
         method_name = json_ast_data[num]['value']
         
         idx = method_name.find('.')
         object_name = None
         if(idx != -1):
             object_name = method_name[0:idx]
             method_name = method_name[idx+1:]
         
         method_name_parameters_type = []
         method_name_parameters = []
         if(json_ast_data[num].get('children') != None):
           child_num = json_ast_data[num]['children']
           #print('child_num = ', child_num)
           for j in child_num:
            if(json_ast_data[j].get('value') != None):
               method_name_parameter = json_ast_data[j]['value'].replace("\"","")
               method_name_parameters.append(method_name_parameter)
               if(dict_variable.get(method_name_parameter) != None):
                   method_name_parameters_type.append(dict_variable[method_name_parameter])
               else:
                   method_name_parameter_type = method_name_parameter[0].swapcase()+method_name_parameter[1:]
                   method_name_parameters_type.append(method_name_parameter_type)
         #print('method_name = ', method_name)
         #print('method_type = ', method_name_parameters_type)
         return_type = "void"
         line_str = source_code.split("\n")
         for j in range(len(line_str)):
             str_return = r'(.*) (.*)=' + method_name + "[(]"
             match_return = re.search(str_return,line_str[j])
             if(match_return):
                #print('mmmmmatch_return = ', match_return.group())
                return_type = match_return.group(1).replace(" ", "")
         #print('object_name = ', object_name)
         if(object_name != None):
             
           if(dict_variable.get(object_name) != None):
             class_type = dict_variable[object_name]
             
           else:
            try:
             class_type = object_name[0].swapcase() + object_name[1:]
            except:
             class_type = object_name.swapcase()
            source_code = source_code + '\nclass ' + class_type + '{}'
             
           str_insert_class = 'class '+ class_type +'{'
           index_insert_class = source_code.find(str_insert_class) + 7 + len(class_type)
           print('index_insert_class  = ', index_insert_class)
           pre_string = source_code[0:index_insert_class]
           nxt_string = source_code[index_insert_class:]
           pre_string = pre_string + '\n'+ "public " + "int" + " " + method_name + "("
           for idd in range(len(method_name_parameters_type)):
                 pre_string = pre_string + method_name_parameters_type[idd] + " " + method_name_parameters[idd] + ","
           if(len(method_name_parameters_type) > 0):
                 pre_string = pre_string[:-1]
           pre_string = pre_string + "){return 1;}"
           source_code = pre_string + nxt_string
    
         else:
             pos_new_function = source_code.find('\n')
             
             pre_string = source_code[0:pos_new_function]
             nxt_string = source_code[pos_new_function:]
             
             pre_string = pre_string+"\n" + "public " + "int" + " "+ method_name + "("
             for idd in range(len(method_name_parameters_type)):
                 pre_string = pre_string + method_name_parameters_type[idd] + " " + method_name_parameters[idd] + ","
             if(len(method_name_parameters_type) > 0):
                 pre_string = pre_string[:-1]
             pre_string = pre_string + "){return 1;}"
             source_code = pre_string + nxt_string
             
    if(json_ast_data[num].get('children') != None):
      for i in json_ast_data[num]['children']:
        dfs1(i)
    #return source_code
def write_file(file_name, code):
    f = open(file_name, 'w')
    f.write(code)
if __name__ == '__main__':
   
    
    with open('test1.json', 'r') as fp:
        json_data = json.load(fp)
        for i in range(16, 20):
            dict_variable = {} 
            dict_node = {}  
            dict_class = {}  
            
            source_code = json_data[str(i)]['source_code']
            index_class = {}
            code = source_code.replace("\n", "")
            source_code = 'public class XXX{\n' + source_code + '}' 
            
            str_piece = source_code.split('\n')
            write_file('source.code', code)
            get_ast('source.code', 'ast.json')
            with open('ast.json', 'r') as fpj:
                json_ast_data = json.load(fpj)
                dfs(0)
                print(dict_node)
                code_line = source_code.split('\n')
                pattern_new = re.compile(r'(.*) [=]new (.*)', re.S)
                for j in range(len(code_line)):
                    match_new = re.search(r'(.*) (.*)[=]new (.*)',code_line[j])
                    if(match_new):
                        class_type = match_new.group(1).replace(" ","")
                        obj = match_new.group(2)
                        method_type = match_new.group(3)
                        dict_class[class_type] = 1
                        dict_variable[obj] = class_type
                for key in dict_node.keys():
                    if(key.find('ReferenceType') != -1):
                        dict_class[dict_node[key]] = 1
                    if(key.find('MemberReference') != -1):
                        variable = dict_node[key]
                        
                        idx = variable.find('.')
                        if(idx != -1):  
                            variable_name = variable[0:idx]
                            variable_method = variable[idx+1:]
                            print('variable = ', variable_name, 'method = ', variable_method)
                            variable = variable_name
                        if(dict_variable.get(variable) == None):
                          try:
                              class_variable = variable[0].swapcase()+variable[1:]
                          except:
                            class_variable = variable.swapcase()  
                          source_code = source_code + '\nclass ' + class_variable + '{}'
                          pos_new_function = source_code.find('\n')
             
                          pre_string = source_code[0:pos_new_function]
                          nxt_string = source_code[pos_new_function:]
              
                          pre_string = pre_string+"\n" + "private " + class_variable + " " + variable + ";"
                          source_code = pre_string + nxt_string
                          dict_variable[variable] = class_variable
                for key in dict_class.keys():
                    source_code = source_code + '\nclass '+ key + '{}'    
                dfs1(0)
                #source_code = code_merge_function
                
                print("final:\n", source_code)
            
                
     
        
