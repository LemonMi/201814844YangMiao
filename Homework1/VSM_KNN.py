import os
from textblob import TextBlob
from textblob import WordList
from textblob import Word
import collections
from nltk.corpus import stopwords
import pickle
import random
from scipy.sparse import csr_matrix
import math
from sklearn.neighbors import BallTree
import numpy as np
import time

def get_file_path(dir_path):
    file_path = []
    for root, _, fnames in os.walk(dir_path):
        for fname in fnames:
            tmp_file_path = os.path.join(root, fname)
            file_path.append(tmp_file_path)
    return file_path

def save_file_dict(file_path, save_file_dict_path):
    # file_dict = {}
    # file_dict_keys = []
    stop_words = stopwords.words('english')
    count = 0
    for fname in file_path:
        with open(fname, 'r', encoding='ISO-8859-1') as f:
            # tokenization / normalization
            s = f.read().lower()
            s = s.replace('/', ' ')
            s = s.replace('-', ' ')
            w = TextBlob(s).words

            clean_w_list = []
            for word in w:
                # stemming (w = w.stem() another way)
                word = Word(word)
                word = word.lemmatize()
                word = Word(word)
                word = word.lemmatize("v")
                # stopwords (nltk.download("stopwords") download the dataset)
                if(word not in stop_words and (word not in ['\'s', '\'ll', '\'t'])):
                    clean_w_list.append(word)

            clean_w_dict = dict(collections.Counter(clean_w_list))
            
            path_list = fname.split('\\')
            save_dict_name = path_list[-2] + '_' + path_list[-1]
            save_dict_path = save_file_dict_path + '/' + save_dict_name + '.txt'
            save_dict(clean_w_dict, save_dict_path)
            # file_dict[save_dict_name] = clean_w_dict
            # file_dict_keys.append(save_dict_name)

            print("The %dth file finished."%(count))
            count += 1
    # return file_dict, file_dict_keys

def get_training_data(dir_path, file_dict_path, test_rate = 0.2):
    train_file_path = []
    test_file_path = []
    for root, _, fnames in os.walk(dir_path):
        total_num = len(fnames)
        test_num = int(total_num * test_rate)
        index_list = range(total_num)
        test_index = random.sample(index_list, test_num)
        train_index = list(set(index_list) - set(test_index))
        for index in test_index:
            fname = fnames[index]
            path_list = root.split('\\')
            tmp_path = file_dict_path + '/' + path_list[-1] + '_' + fname + '.txt'
            test_file_path.append(tmp_path)
        for index in train_index:
            fname = fnames[index]
            path_list = root.split('\\')
            tmp_path = file_dict_path + '/' + path_list[-1] + '_' + fname + '.txt'
            train_file_path.append(tmp_path)
    return (train_file_path, test_file_path)

def generate_dict_from_filedict(file_dict_path):
    global_dict = {}
    count = 0
    for fname in file_dict_path:
        tmp_file_dict = load_dict(fname)
        for key in tmp_file_dict:
            if(key in global_dict.keys()):
                global_dict[key] += 1
            else:
                global_dict[key] = 1
        # print("The %dth file finished."%(count))
        count += 1
    return global_dict
    
def save_dict(dict_name, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dict_name, f)
def load_dict(load_path):
    with open(load_path, 'rb') as f:
        tmp_load_dict = pickle.load(f)
    return tmp_load_dict

def get_vector_from_data(data, train_data_num, global_dict_filtered, global_dict_filtered_index):
    file_num = len(data)
    dict_num = len(global_dict_filtered)

    # construct the train sparse vector
    col = []
    row = []
    value = []
    for idx, path in enumerate(data):
        tmp_file_dict = load_dict(path)
        for key in tmp_file_dict:
            if(key in global_dict_filtered.keys()):
                row.append(idx)
                col.append(global_dict_filtered_index[key])
                # caculate the TF(Sub-linear TF scaling)
                tf = 1 + math.log(tmp_file_dict[key])
                # caculate the IDF
                idf = math.log(train_data_num / global_dict_filtered[key])
                # caculate the TF * IDF
                value.append(tf * idf)
    vector = csr_matrix((value, (row, col)), shape=(file_num, dict_num))
    for i in range(file_num):
        norm = pow(vector[i].power(2).sum(), 1/2)
        vector[i] /= norm
    return vector

if __name__=="__main__":

    #TODO:generate the file dict
    # print("The file dict generation has began")
    # begin = time.time()
    # dir_path = './20news-18828'
    # save_file_dict_path = './file_dict_folder'
    # file_path = get_file_path(dir_path)
    # get_file_dict(file_path, save_file_dict_path)
    # end = time.time()
    # print("The file dict generation has finished: it takes %fs"%(end - begin))

    #TODO generate training data/ testing data
    dir_path = './20news-18828'
    save_file_dict_path = './file_dict_folder'
    print("The training/tesing data generation has began")
    begin = time.time()
    save_train_path = './training_path.txt'
    save_test_path = './testing_path.txt'
    train_data, test_data = get_training_data(dir_path, save_file_dict_path)
    with open(save_train_path, 'w') as f:
        for path in train_data:
            f.write(path + '\n')
    with open(save_test_path, 'w') as f:
        for path in test_data:
            f.write(path + '\n')
    end = time.time()
    print("The training/testing data generation has finished: it takes %fs"%(end - begin))

    #TODO generate the global dict
    print("The filtered global dict generation has began")
    begin = time.time()

    global_dict = generate_dict_from_filedict(train_data)
    # generate the filtered global dict
    FILTER_LOW_FREQUENCY = 15
    global_dict_filtered = {}
    for key in global_dict:
        if global_dict[key] >= FILTER_LOW_FREQUENCY:
            global_dict_filtered[key] = global_dict[key]
    save_dict(global_dict_filtered, './global_dict_filtered.txt')
    # map filtered global dict to index
    global_dict_filtered_index = {}
    index = 0
    for key in global_dict_filtered:
        global_dict_filtered_index[key] = index
        index += 1
    
    end = time.time()
    print("The filtered global dict generation has finished: it takes %fs"%(end - begin))

    #TODO caculate the KNN
    train_data_num = len(train_data)

    print("The train vector construction has began")
    begin = time.time()
    train_vector = get_vector_from_data(train_data, train_data_num, global_dict_filtered, global_dict_filtered_index)
    end = time.time()
    print("The train vector construction has finished: it takes %fs"%(end - begin))
    
    print("The test vector construction has began")
    begin = time.time()
    test_vector = get_vector_from_data(test_data, train_data_num, global_dict_filtered, global_dict_filtered_index)
    end = time.time()
    print("The test vector construction has finished: it takes %fs"%(end - begin))

    k = 5
    print("The Ball Tree construction has began")
    begin = time.time()
    ball_tree = BallTree(train_vector.toarray(), leaf_size=2, metric='euclidean')
    end = time.time()
    print("The Ball Tree construction has finished: it takes %fs"%(end - begin))

    train_label = []
    for i in range(len(train_data)):
        tmp_train_label = train_data[i].split('/')[-1].split('_')[0]
        train_label.append(tmp_train_label)
    train_label = np.array(train_label)

    print("The Ball Tree Query has began")
    begin = time.time()

    total_num = 0
    true_num = 0
    for i in range(len(test_data)):
        tmp_test_label = test_data[i].split('/')[-1].split('_')[0]
        dist, indices = ball_tree.query(test_vector[i].toarray(), k=5)
        indices = np.array(indices[0])
        vote_list = train_label[indices]
        vote_dict = {}
        weight = 1/ (dist[0]*dist[0])
        for idx, key in enumerate(vote_list):
            if(key in vote_dict.keys()):
                vote_dict[key] += weight[idx]
            else:
                vote_dict[key] = weight[idx]
        # vote_dict = collections.Counter(vote_list)
        predict_label = max(zip(vote_dict.values(), vote_dict.keys()))
        print("%d: test_label:%s vote_status:%s predict_label:%s"%(total_num, tmp_test_label, vote_list, predict_label))
        
        total_num += 1
        if(predict_label[1] == tmp_test_label):
            true_num += 1
    print("true_num:%d total_num:%d accuracy:%f"%(true_num, total_num, float(true_num) / total_num))
    end = time.time()
    print("The Ball Tree Query has finished: it takes %fs"%(end - begin))

    # print("The KNN has began")
    # begin = time.time()
    # test_file_num = len(test_data)
    # train_file_num = len(train_data)
    # total_num = 0
    # true_num = 0
    # for i in range(test_file_num):
    #     tmp_test_file = test_data[i]
    #     tmp_test_label = tmp_test_file.split('/')[-1].split('_')[0]
    #     tmp_test_vector = test_vector[i]
    #     norm_tmp_test_vector = pow(tmp_test_vector.power(2).sum(), 1/2)
        
    #     vote_label = []
    #     vote_distance = []
    #     for j in range(train_file_num):
    #         tmp_train_file = train_data[j]
    #         tmp_train_label = tmp_train_file.split('/')[-1].split('_')[0]
    #         tmp_train_vector = train_vector[j]
    #         norm_tmp_train_vector = pow(tmp_train_vector.power(2).sum(), 1/2)

    #         distance = tmp_train_vector.multiply(tmp_test_vector).sum() / (norm_tmp_test_vector * norm_tmp_train_vector)
    #         vote_label.append(tmp_train_label)
    #         vote_distance.append(distance)

    #     vote_distance_np = np.array(vote_distance)
    #     vote_distance_np = -1 * vote_distance_np
    #     vote_index = np.argpartition(vote_distance_np, k)[:k]

    #     vote_label = np.array(vote_label)
    #     vote_list = vote_label[vote_index]
    #     vote_dict = collections.Counter(vote_list)

    #     predict_label = max(zip(vote_dict.values(), vote_dict.keys()))
    #     print("%d: test_label:%s vote_status:%s predict_label:%s"%(total_num, tmp_test_label, vote_list, predict_label))
        
    #     total_num += 1
    #     if(predict_label[1] == tmp_test_label):
    #         true_num += 1

    # print("true_num:%d total_num:%d accuracy:%f"%(true_num, total_num, float(true_num) / total_num))

    end = time.time()
    internal = end - begin
    print("The KNN has finished: it takes %fs"%(internal))