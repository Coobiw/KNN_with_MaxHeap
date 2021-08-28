import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from KNN import K_Nearest_Neighbor

def read_data(path,separate):
    '''
    :param path: the path of the data file
    :param separate: the separate symbol of the file
    :return: data and target(label)
    '''

    dataset = pd.read_csv(path,sep = separate)
    # print(type(dataset))
    col = dataset.columns.values.tolist()

    # print(dataset)
    # print(col)

    data_X = np.array(dataset[col[1:3]])
    data_Y = np.ones((data_X.shape[0],1))
    for i in range(data_X.shape[0]):
        if dataset[col[-1]][i] == 'setosa':
            data_Y[i][0] = 0
    return data_X,data_Y

def fig_plot(data_X,data_Y,axs,flag,marker ='o'):
    '''

    :param data_X: data
    :param data_Y: target
    :param axs: 作图轴
    :param flag: 选择哪个颜色集合（事实上是用于区分训练和预测，为False则是训练，否则为预测）
    :param marker: 散点图的点形状 默认为圆形
    :return:
    :actual function: 作图
    '''
    if flag==False:
        colors = ['red','blue']
        for i in range(data_X.shape[0]):
            axs.set_xlim(0, 1)
            axs.set_ylim(0, 1)
            axs.scatter(data_X[i,0],data_X[i,1],color=colors[int(data_Y[i][0])],marker=marker)

    else:
        colors = ['pink','green']
        for i in range(data_X.shape[0]):
            ## 输出大于0.5则为1，小于0.5则为0（用round函数四舍五入实现）
            axs.set_xlim(0, 1)
            axs.set_ylim(0, 1)
            axs.scatter(data_X[i,0],data_X[i,1],color=colors[int(round(data_Y[i][0]))],marker=marker)


def dataset_split(data_X,data_Y,val_ratio):
    '''

    :param data_X: data
    :param data_Y: target
    :param val_ratio: the ratio of validation set in all the data
    :return: the data and target of training set and validation set

    :actual function: random split the training set and validation set
    '''
    m = data_X.shape[0]
    val_num = int(val_ratio * m)
    train_num = m - val_num
    total_list = [i for i in range(m)]
    train_index = random.sample(total_list,train_num)
    train_index.sort()
    for each in train_index:
        total_list.remove(each)
    train_X = np.zeros((train_num,data_X.shape[1]))
    train_Y = np.zeros((train_num,1))
    val_X = np.zeros((val_num,data_X.shape[1]))
    val_Y = np.zeros((val_num,1))

    # print(train_index)
    for i,each in enumerate(train_index):
        train_X[i] = data_X[each]
        train_Y[i] = data_Y[each]

    for i,each in enumerate(total_list):
        val_X[i] = data_X[each]
        val_Y[i] = data_Y[each]

    return train_X,train_Y,val_X,val_Y


def main():
    # print(read_data.__doc__)
    # print(help(fig_plot))
    # print(dataset_split.__doc__)


    # load the data
    path = 'iris_two_classes.csv'
    data_X,data_Y = read_data(path,separate=',')


    # split the train and the validation set
    train_X,train_Y,val_X,val_Y = dataset_split(data_X,data_Y,0.5)

    # initialize the knn
    knn = K_Nearest_Neighbor(train_X,train_Y,val_X,val_Y)

    # train the model (knn is lazy-training model)
    knn.model_train()

    # validation of the model
    print(knn.model_validation(1))
    print(knn.model_score(1))

    # plot the training set,validation set,the whole data
    plt.figure()
    ax1 = plt.axes()
    plt.figure()
    ax2 = plt.axes()
    plt.figure()
    ax3 = plt.axes()

    fig_plot(knn.train_X,knn.train_Y,ax1,False)
    fig_plot(knn.val_X,knn.val_Y,ax2,True)
    fig_plot(np.concatenate((knn.train_X,knn.val_X),axis=0),np.concatenate((knn.train_Y,knn.val_Y),axis=0),ax3,False)
    ax4 = knn.choose_the_best_K()
    plt.show()

if __name__ == "__main__":
    main()