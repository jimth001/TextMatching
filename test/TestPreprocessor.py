# coding:utf-8
from Preprocess.Preprocessor import Preprocessor

path = "E:\\Learning\\研一\\文本匹配\\NLPCC2017-OpenDomainQA\\evaltool\\dbqa\\sample.QApair.txt"
x, y = Preprocessor.load_data(path)
print(x)
print(y)
x_vec, dict = Preprocessor.seg_and_2_int(x)
batch_iter = Preprocessor.batch_iter(x_vec, y, batch_size=2)
for x_, y_, mask in batch_iter:
    print(x_)
    print(y_)
    print(mask)
balanced_x,balanced_y=Preprocessor.get_balanced_data(x_vec,y)
data = Preprocessor.generate_train_and_cross_validation(x_vec, y, n_fold=2)
x_train, y_train, x_val, y_val = data.__next__()
print("hhhhhhhhhhhhhhhhhhhhhhhhhhhh")
print(x_train)
print(y_train)
print(x_val)
print(y_val)
for x_train, y_train, x_val, y_val in data:
    print('......................')
    print(x_train)
    print(y_train)
    print(x_val)
    print(y_val)
