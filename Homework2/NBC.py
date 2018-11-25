import os 
import pickle
import numpy as np
import time

def save_dict(dict_name, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dict_name, f)
def load_dict(load_path):
    with open(load_path, 'rb') as f:
        tmp_load_dict = pickle.load(f)
    return tmp_load_dict


print("Initialization has begin.")
begin = time.time()

#TODO: count file
total_count = 0 #文件总数
class_name = [] #类别名称
class_file_count = {} #每个类别的文件总数
with open('./training_path.txt', 'r') as f:
    for line in f:
        line = line.strip()
        file_name = line.split('/')[-1]
        tmp_class_name = file_name.split('_')[0]

        total_count += 1
        if(tmp_class_name not in class_name):
            class_name.append(tmp_class_name)
        if(tmp_class_name not in class_file_count):
            class_file_count[tmp_class_name] = 1
        else: 
            class_file_count[tmp_class_name] += 1        

#TODO: map filtered global dict to index
global_dict = load_dict('./global_dict_filtered.txt')
global_dict_index = {}
index = 0
for key in global_dict:
    global_dict_index[key] = index
    index += 1
term_count = len(global_dict_index)

#TODO: map class name to index
class_name_index = {}
index = 0
for name in class_name:
    class_name_index[name] = index
    index += 1
class_count = len(class_name_index)

#TODO: construct the reference matrix with size(class_count, term_count)
ref_matrix = np.zeros([class_count, term_count])
with open('training_path.txt', 'r') as f:
    for line in f:
        line = line.strip()
        tmp_file_dict = load_dict(line)
        file_name = line.split('/')[-1]
        tmp_class_name = file_name.split('_')[0]

        for key in tmp_file_dict:
            if(key in global_dict):
                class_index = class_name_index[tmp_class_name]
                term_index = global_dict_index[key]
                ref_matrix[class_index][term_index] += tmp_file_dict[key]
end = time.time()
print("Initialization has finished.It takes %fs"%(end - begin))


real_num = 0
total_num = 0
with open('testing_path.txt', 'r') as f:
    for line in f:
        line = line.strip()
        tmp_file_dict = load_dict(line)
        file_name = line.split('/')[-1]
        real_name = file_name.split('_')[0]

        p = np.zeros(class_count)

        # 多项式模型 laplace平滑
        for class_idx in range(class_count):
            tmp_class_name = class_name[class_idx]
            tmp_class_file_count = class_file_count[tmp_class_name]
            p_c = float(tmp_class_file_count) / total_count
            
            p_t_c = 0
            total_term = np.sum(ref_matrix[class_idx])
            for key in tmp_file_dict:
                if(key in global_dict.keys()):
                    index = global_dict_index[key]
                    tmp_term = ref_matrix[class_idx][index]
                    p_t_c += tmp_file_dict[key] * np.log(float(tmp_term + 1) / (total_term + term_count))

            p[class_idx] = p_t_c + np.log(p_c) 

        max_index = np.argmax(p)
        predict_name = class_name[max_index]

        if(predict_name == real_name):
            real_num += 1
        total_num += 1
        print("true label:%s predict_label:%s"%(real_name, predict_name))

print(float(real_num) / total_num)
                


