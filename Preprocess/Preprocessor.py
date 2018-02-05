# coding:utf-8
from .WordDictionary import WordDictionary
import numpy as np
from .Tokenizer import Tokenizer
import codecs

class Preprocessor:
    @staticmethod
    def load_data(data_path):
        # 数据格式:csv
        # todo
        x_data = []
        y_data = []
        has_y = True
        file = codecs.open(data_path, 'r', encoding='utf-8')
        for line in file:
            strs = line.strip('\n').split('\t')
            x_data.append(strs[0] + strs[1])  ##文本匹配有两个文本。暂时拼到一起。后面应该改成过两个RNN的
            if has_y:
                try:
                    y_data.append(strs[2])  # 注意target是str
                except IndexError:
                    y_data = None
                    has_y = False
        return x_data, y_data

    @staticmethod
    def target_2_one_hot(y_data,target_dict=None):  # 适用于multi-class classification。允许target是各种形式，比如“类别1，类别2”
        is_new_dict=False
        if target_dict==None:
            target_dict = WordDictionary()
            is_new_dict=True
        y_index = []
        for tar in y_data:
            index = target_dict.get_index(tar)
            if index is None:
                if is_new_dict:
                    target_dict.add_word(tar)
                    index = target_dict.get_index(tar)
                else:
                    raise ValueError("未知的类别："+str(tar))
            y_index.append(index)
        size = target_dict.get_size()
        y_new = []
        for index in y_index:
            tmp = [0] * size
            tmp[index] = 1
            y_new.append(tmp)
        return y_new, target_dict

    @staticmethod
    def seg_and_2_int(x_data, word_dict=None):
        tokenizer = Tokenizer()
        new_x = []
        if word_dict is None:
            dict = WordDictionary()
            for sen in x_data:
                sen_vec = []
                seg_rst = tokenizer.parser(sen).split(' ')
                for word in seg_rst:
                    index = dict.get_index(word)
                    if index is None:
                        dict.add_word(word)
                        sen_vec.append(dict.get_index(word))
                    else:
                        sen_vec.append(index)
                new_x.append(sen_vec)
            word_dict = dict
        else:
            for sen in x_data:
                sen_vec = []
                seg_rst = tokenizer.parser(sen)
                for word in seg_rst:
                    index = word_dict.get_index(word)
                    if not index is None:  # 如果用已有词典，那么出现未登录词的时候丢弃这个词
                        sen_vec.append(index)
                new_x.append(sen_vec)
        return new_x, word_dict

    @staticmethod
    def save_preprocessed_data(data,target=None):
        #todo
        pass

    @staticmethod
    def load_preprocessed_data(data_path):
        #todo
        pass


    @staticmethod
    def get_balanced_data(data, target):
        if len(data) != len(target):
            raise ValueError("x和y长度不一致")
            # todo 过采样得到均衡数据集
        target_num_counter = {}
        target_data_dict = {}
        new_data = [x for x in data]
        new_target = [x for x in target]
        for i in range(len(target)):
            if target[i] in target_num_counter:
                target_num_counter[target[i]] += 1
            else:
                target_num_counter[target[i]] = 1
            if target[i] in target_data_dict:
                target_data_dict[target[i]].append(i)
            else:
                target_data_dict[target[i]] = [i]
        max = 0
        for key in target_num_counter.keys():
            if target_num_counter[key] > max:
                max = target_num_counter[key]
        for key in target_data_dict.keys():
            list = target_data_dict[key]
            length = len(list)
            for i in range(max - length):
                random_index = np.random.randint(0, length)
                new_data.append(data[list[random_index]])
                new_target.append(target[list[random_index]])
        return new_data, new_target

    @staticmethod
    def batch_iter(x, y, batch_size=64, padding=0):  # 生成批次数据。x中数据可以不等长，会padding成一样的
        if y is None or x is None:
            raise ValueError("x or y is None")
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = [x[i] for i in indices]
        y_shuffle = [y[i] for i in indices]
        mask = [len(x_shuffle[i]) for i in range(len(x_shuffle))]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            rx = x_shuffle[start_id:end_id]
            ry = y_shuffle[start_id:end_id]
            rm = mask[start_id:end_id]
            max_len = max(rm)
            for list in rx:
                if len(list) < max_len:
                    list += [padding] * (max_len - len(list))
            yield rx, ry, rm

    @staticmethod
    def generate_train_and_cross_validation(x, y, n_fold=4):  # 根据x，y划分n-fold交叉验证的数据集
        if len(x) != len(y):
            raise ValueError("x和y长度不一致")
        val_len = int(len(x) / n_fold)
        start_id = 0
        indices = np.random.permutation(np.arange(len(x)))
        x_shuffle = [x[i] for i in indices]
        y_shuffle = [y[i] for i in indices]
        for i in range(n_fold):
            x_train = [x_shuffle[r] for r in range(len(x)) if (r < start_id or r > start_id + val_len)]
            y_train = [y_shuffle[r] for r in range(len(x)) if (r < start_id or r > start_id + val_len)]
            x_val = [x_shuffle[r] for r in range(len(x)) if (r >= start_id and r <= start_id + val_len)]
            y_val = [y_shuffle[r] for r in range(len(x)) if (r >= start_id and r <= start_id + val_len)]
            start_id = start_id + val_len + 1
            yield x_train, y_train, x_val, y_val
