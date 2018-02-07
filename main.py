# coding:utf-8
from BaseTensorFlow import RNNBasedModel
from Preprocess.Preprocessor import Preprocessor

if __name__ == "__main__":
    print("loading config......")
    config = RNNBasedModel.RNNConfig("my_rnn","matching")
    print("starting build nn model")
    rnn = RNNBasedModel.RNNModel(config)
    rnn.train_onehotkey("./training_testing/nlpcc-iccpol-2016.dbqa.training-data",data_preprocessed=True,release_resources=True, weight_balanced=True)
    print("training finished")
    rnn.evaluate_onehotkey("./training_testing/nlpcc-iccpol-2016.dbqa.testing-data",init_nn=False,load_model=True,data_preprocessed=False)
    '''x_test, y_test = Preprocessor.load_data("./training_testing/nlpcc-iccpol-2016.dbqa.testing-data")
    rnn.load()
    x_test, y = Preprocessor.seg_and_2_int(x_test, rnn.word_dict)
    y_test, _ = Preprocessor.target_2_one_hot(y, rnn.target_dict)
    rst = rnn._evaluate_without_predict_result(x_test, y_test)
    print("loss and acc is:")
    print(rst)
    '''
    print("all work finished")
