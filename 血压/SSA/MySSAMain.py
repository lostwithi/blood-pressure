import tensorflow as tf
import numpy as np
import pandas as pd
from MyFun.SSAFunction import function
import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


"""=============读取训练集================="""
f_train = open("remoteTrainData.csv", 'r', encoding='UTF-8')
train_datalael = pd.read_csv(f_train, header=None, error_bad_lines=False)
train_datalael = train_datalael.iloc[:, 1:]
train_datalael = train_datalael.values
train_datalael = train_datalael.astype(np.float)


"""SBP的"""

# feaList = [2, 34, 1, 9, 3, 5, 12, 16, 20, 21, 25, 6, 37, 22, 15, 19, 32, 4, 11, 10, 33, 27, 35, 30, 36, 38, 13, 28, 39,
#            14, 29, 18, 24]
# train_data = train_datalael[:, feaList]
# train_label = train_datalael[:, -2]

"""DBP的"""
feaList = [1, 7, 16, 10, 5, 21, 4, 18, 14, 36, 0, 37, 33, 13, 27, 20, 11, 12, 23, 34, 9, 39, 2, 25, 6, 22, 3, 24]
train_data = train_datalael[:, feaList]
train_label = train_datalael[:, -1]

"""BP模型参数"""
num_classes = 1
input_size = len(train_data[0])
hidden_units_size = 2 * input_size + 1
batch_size = 64
training_iterations = 1000
learning_rate = 0.01



"""设置SSA种群信息"""

NIND = 50  # 种群规模
Dim = input_size * hidden_units_size + hidden_units_size + hidden_units_size * num_classes + num_classes  # 初始化Dim（决策变量维数
lb = -1
ub = 1
Max_iteration = 50






"""=================初始化种群============================"""
pop = []
with tf.Session() as sess:
    for i in range(NIND):

        print(i)

        W1 = tf.Variable(tf.random_normal([input_size, hidden_units_size], stddev=0.1))
        B1 = tf.Variable(tf.random_normal([1, hidden_units_size], stddev=0.1))
        W2 = tf.Variable(tf.random_normal([hidden_units_size, num_classes], stddev=0.1))
        B2 = tf.Variable(tf.random_normal([1, num_classes], stddev=0.1))


        init = tf.global_variables_initializer()
        sess.run(init)

        W1_num = W1.eval(sess)
        W1_num = np.ravel(W1_num)

        B1_num = B1.eval(sess)
        B1_num = np.ravel(B1_num)

        W2_num = W2.eval(sess)
        W2_num = np.ravel(W2_num)

        B2_num = B2.eval(sess)
        B2_num = np.ravel(B2_num)

        weight = []
        weight.extend(W1_num.tolist())
        weight.extend(B1_num.tolist())
        weight.extend(W2_num.tolist())
        weight.extend(B2_num.tolist())

        pop.append(weight)

# 关闭会话，防止资源泄漏
sess.close()
pop = np.array(pop)

fun = function(input_size,hidden_units_size,num_classes,train_data,train_label,batch_size,training_iterations,learning_rate,pop)
[fMin,bestX,SSA_curve] = fun.SSA(NIND,Max_iteration,lb,ub,Dim,"F1")
print()
print(['最优值为：', fMin])
print(['最优变量为：', bestX])

weight = pd.DataFrame(bestX)
weight.to_csv("weight.csv", header=False, index=False)



print("寻优结束")