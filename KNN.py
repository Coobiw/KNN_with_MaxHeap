import numpy as np
import matplotlib.pyplot as plt
import math
import random
from max_heap import big_root_heap
class K_Nearest_Neighbor():
    def __init__(self,train_X,train_Y,val_X,val_Y):
        self.train_X = self.normalization_data(train_X)
        self.train_Y = train_Y

        # self.weight = np.zeros((data_X.shape[1],1)) # 零初始化
        self.weight = np.random.randn(train_X.shape[1], 1)  # 标准正态分布初始化
        self.bias = random.random() # bias随机初始化到[0,1]

        self.m = train_X.shape[0]
        self.val_X = self.normalization_data(val_X)
        self.val_Y = val_Y

    def normalization_data(self,data_X):
        '''

        :param data_X: the data which need to be scaled to [0,1]
        :return:
        :actual function: scale the data to [0,1]
        '''
        eps = 0.000001 #防止分母为0
        dimension = data_X.shape[1]
        for i in range(0,dimension):
            min_val = np.min(data_X[:,i])
            max_val = np.max(data_X[:,i])
            data_X[:,i] = (data_X[:,i]-min_val)/(max_val-min_val+eps)

        return data_X

    def distance_computation(self,vector1,vector2):
        return np.linalg.norm((vector1-vector2),ord=2)**2

    def model_train(self):
        print("lazy training: just need to load the data when initializing")



    def model_predict(self,K,vector):
        '''

        :param K: hyper para K of K_Nearest_Neighbor
        :param vector: the feature vector need to be predicted
        :return: return the classification prediction of the input feature vector

        :implement detail: use big root heap to do this
        '''
        heap = big_root_heap(maxsize=K)
        for i in range(self.m):
            distance = self.distance_computation(vector,self.train_X[i])
            heap.push({"index":i,"distance":distance})

        counter0 = 0
        counter1 = 0

        # for i,each in enumerate(heap.heap):
        #     if i:
        #         print("index: {}".format(each["index"]),end='  ')
        #         print("distance: {}".format(each["distance"]))

        for i in range(K):
            weight = heap.heap[i]['distance']
            if self.train_Y[heap.heap[i]['index']][0]==0:
                counter0+=1#/weight

            if self.train_Y[heap.heap[i]['index']][0]==1:
                counter1+=1#/weight

        if counter1>=counter0:
            return 1
        else:
            return 0



    def model_validation(self,K):
        '''
        use to do the prediction of the validation set
        '''
        result_list = []
        val_number = self.val_X.shape[0]
        for i in range(val_number):
            result_list.append(self.model_predict(K,self.val_X[i]))
        return result_list

    def model_score(self,K):
        '''
        :return: the score of the model with
                                the hyper parameter which is equal to the input K
        '''
        predictions = self.model_validation(K)
        confusion_matrix = np.zeros((2,2),dtype='uint8')
        for i,_ in enumerate(predictions):
            if int(predictions[i]) == 0 and int(self.val_Y[i][0])==0:
                confusion_matrix[0][0]+=1
            elif int(predictions[i]) == 0 and int(self.val_Y[i][0])==1:
                confusion_matrix[0][1]+=1
            elif int(predictions[i]) == 1 and int(self.val_Y[i][0]) == 1:
                confusion_matrix[1][1]+=1
            elif int(predictions[i]) == 1 and int(self.val_Y[i][0]) == 0:
                confusion_matrix[1][0]+=1
            else:
                try:
                    assert False
                except  AssertionError:
                    print("Classification Error")
        accuracy = (np.diag(confusion_matrix).sum(axis=0))/(confusion_matrix.sum(axis=1).sum(axis=0))

        return accuracy

    def choose_the_best_K(self):
        '''
        use the visualization method to choose the best K of this validation set

        Actually , the larger K will lead to a larger computation cost and space cost.

        The smaller K will lead to worse performance of the model

        '''
        val_number = self.val_X.shape[0]
        kp = np.arange(1,val_number,1)
        score_p = []
        for each in kp:
            score_p.append(self.model_score(each))

        score_p = np.array(score_p)

        plt.figure()
        ax4 = plt.axes()
        ax4.scatter(kp,score_p,color="y")
        return ax4


if __name__ == "__main__":
    X = np.zeros((20,2))
    Y = np.zeros((20,1))
    knn = K_Nearest_Neighbor(X,Y,X,Y)
    print("the train method of knn: \n")
    knn.model_train()
    print(end='\n\n\n')
    print("the document of predict function: \n",knn.model_predict.__doc__,end='\n\n\n')
    print("the document of validation function: \n",knn.model_validation.__doc__,end='\n\n\n')
    print("the document of score function: \n",knn.model_score.__doc__,end='\n\n\n')
    print("the document of choose the best K function: \n", knn.choose_the_best_K.__doc__, end='\n\n\n')
