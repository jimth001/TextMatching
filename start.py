from BaseTensorFlow import RNNBasedModel
from Preprocess.Preprocessor import Preprocessor
config=RNNBasedModel.RNNConfig("my_rnn")
rnn=RNNBasedModel.RNNModel(config)
rnn.train("training&testing\\mini-test.training-data",save_and_quit=True,weight_balanced=True)
print("training finished")
x_test,y_test=Preprocessor.load_data("training&testing\\mini-test.training-data")
rnn.load()
x_test,y=Preprocessor.seg_and_2_int(x_test,rnn.word_dict)
rst=rnn.inference_all(x_test)
print("all work finished")