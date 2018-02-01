from .WordDictionary import WordDictionary
import numpy as np
from .Tokenizer import Tokenizer


class Preprocessor:
    @staticmethod
    def load_data(data_path):
        # 数据格式:csv
        # todo
        x_data = []
        y_data = []
        has_y = True
        file = open(data_path, 'r', encoding='utf-8')
        for line in file:
            strs = line.split('\t')
            x_data.append(strs[0] + strs[1])  ##文本匹配有两个文本。暂时拼到一起。后面应该改成过两个RNN的
            if has_y:
                try:
                    y_data.append(int(strs[2]))
                except IndexError:
                    y_data = None
                    has_y = False
        return x_data, y_data

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
                    index = dict.get_index(word)
                    if not index is None:  # 如果用已有词典，那么出现未登录词的时候丢弃这个词
                        sen_vec.append(index)
                new_x.append(sen_vec)
        return new_x, word_dict

    @staticmethod
    def get_balanced_data(data, target):
        if len(data) != len(target):
            raise ValueError("x和y长度不一致")
            # todo 过采样得到均衡数据集

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
                    list = list + [padding] * (max_len - len(list))
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
