# coding:utf-8
from BaseTensorFlow import RNNBasedModel
from Preprocess.Preprocessor import Preprocessor

if __name__ == "__main__":
    print("loading config......")
    config = RNNBasedModel.RNNConfig("my_rnn","matching")
    print("starting build nn model")
    rnn = RNNBasedModel.RNNModel(config)
    #nlpcc-iccpol-2016.dbqa.training-data
    #mini-test.training-data
    rnn.train_onehotkey("./training_testing/nlpcc-iccpol-2016.dbqa.training-data",data_preprocessed=False,release_resources=True, weight_balanced=True)
    print("training finished")
    rnn.evaluate_onehotkey("./training_testing/nlpcc-iccpol-2016.dbqa.testing-data",init_nn=False,load_model=True,data_preprocessed=False)
    print("all work finished")
