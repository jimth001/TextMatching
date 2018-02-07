class Metrics:
    @staticmethod
    def get_mrr(right_answer,predicted_ranking):
        #predict_ran
        for i in len(right_answer):#一个query的正确的response只有一个
            index=right_answer