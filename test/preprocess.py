from Preprocess.Preprocessor import Preprocessor

word_dict,target_dict=Preprocessor.preprocess_and_save("../training_testing/nlpcc-iccpol-2016.dbqa.training-data","../training_testing/",weight_balanced=True)
Preprocessor.preprocess_and_save("../training_testing/nlpcc-iccpol-2016.dbqa.testing-data","../training_testing/",name="test",weight_balanced=True,word_dict=word_dict,target_dict=target_dict)
##target_dict有bug，待修复，201802052251