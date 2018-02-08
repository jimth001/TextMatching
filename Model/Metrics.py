class Metrics:
    @staticmethod
    def stringlize(a):
        rst=[]
        for line in a:
            rst.append(",".join([str(x) for x in line]))
        return rst

    @staticmethod
    def calculate_mrr(input_data,pred,label,target_dict):
        #only for matching task,label is 2-dim vector represents 0 or 1,input_data contains query and response
        query=Metrics.stringlize(input_data[0])
        response=input_data[1]
        dict_response_ranking={}
        if not(len(query)==len(response) and len(query)==len(pred) and len(query)==len(label)):
            raise ValueError("query,response,pred and label are not in same length")
        for i in range(len(query)):
            if query[i] in dict_response_ranking:
                rank_list=dict_response_ranking[query[i]]
                if Metrics.one_hot_vec2index(label[i])==target_dict.get_index('1'):#如果是正确答案
                    rank_list.insert(0,pred[i][target_dict.get_index('1')])
                else:
                    rank_list.append(pred[i][target_dict.get_index('1')])
            else:
                dict_response_ranking[query[i]]=[pred[i][target_dict.get_index('1')]]
        mrr=0
        for key in dict_response_ranking.keys():
            rank_list=dict_response_ranking[key]
            counter=1
            for rank in rank_list:
                if rank>rank_list[0]:
                    counter+=1
            mrr+=1/counter
        mrr=mrr/len(dict_response_ranking)
        return mrr

    @staticmethod
    def one_hot_vec2index(a):
        for i in range(len(a)):
            if a[i]==1:
                return i
        raise ValueError("a is not a one_hot_vec")

    @staticmethod
    def is_vector_equal(a,b):
        if len(a)==len(b):
            for i in range(len(a)):
                if a[i]!=b[i]:
                    return False
            return True
        return False

    def __get_mrr(right_answer,predicted_ranking):
        #predict_ran
        for i in len(right_answer):#一个query的正确的response只有一个
            index=right_answer