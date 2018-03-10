# coding:utf-8
from Preprocess.Preprocessor import Preprocessor
from Preprocess.WordDictionary import WordDictionary

def print_list(list,seg="-------"):
    print(seg)
    for l in list:
        print(l)

path = "E:\\Learning\\研一\\文本匹配\\NLPCC2017-OpenDomainQA\\evaltool\\dbqa\\sample.QApair.txt"
#word_dict,target_dict=Preprocessor.preprocess_and_save(path,"E:\\",weight_balanced=True)
word_dict=WordDictionary()
word_dict.load("E:\\", "word_dict")
target_dict=WordDictionary()
target_dict.load("E:\\","target_dict")
Preprocessor.preprocess_and_save(path,"E:\\",name="test",word_dict=word_dict,target_dict=target_dict)
train_data,train_target,val_data,val_target,vocab_size=Preprocessor.load_preprocessed_data_for_train("E:\\")
print_list(train_data)
print_list(train_target)
print_list(val_data)
print_list(val_target)
print(vocab_size)

