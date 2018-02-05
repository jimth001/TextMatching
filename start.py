# coding:utf-8
from BaseTensorFlow import RNNBasedModel
from Preprocess.Preprocessor import Preprocessor
config=RNNBasedModel.RNNConfig("my_rnn")
rnn=RNNBasedModel.RNNModel(config)
rnn.train("training&testing\\nlpcc-iccpol-2016.dbqa.training-data",save_and_quit=True,weight_balanced=True)
print("training finished")
x_test,y_test=Preprocessor.load_data("training&testing\\nlpcc-iccpol-2016.dbqa.testing-data")
rnn.load()
x_test,y=Preprocessor.seg_and_2_int(x_test,rnn.word_dict)
y_test,_=Preprocessor.target_2_one_hot(y,rnn.target_dict)
rst=rnn._evaluate_without_predict_result(x_test,y_test)
print("loss and acc is:")
print(rst)
print("all work finished")