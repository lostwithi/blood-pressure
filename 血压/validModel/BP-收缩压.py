import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


"""==================计算指标函数=========================="""
def error_total(pre_values, true_values):
    error_values = []
    for i in range(len(pre_values)):
        error_values.append(pre_values[i] - true_values[i])

    return error_values


def MAE_Fun(pre_values, true_values):
    MAE_value = 0
    In_5 = 0
    In_10 = 0
    In_15 = 0
    for i in range(len(pre_values)):
        ABS = np.abs((pre_values[i] - true_values[i]))
        MAE_value += np.abs((pre_values[i] - true_values[i]))
        if (ABS <= 5):
            In_5 += 1
        if (ABS <= 10):
            In_10 += 1

        if (ABS <= 15):
            In_15 += 1
    print("5mmHg个数：",str(In_5))
    print("10mmHg个数：", str(In_10))
    print("15mmHg个数：", str(In_15))
    return MAE_value / len(pre_values), In_5 / len(pre_values), In_10 / len(pre_values), In_15 / len(pre_values)




def modelValid(data,label):
    tf.reset_default_graph()




    # saver = tf.train.Saver()
    # 创建保存模型的目录

    with tf.Session() as sess:
        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # 使用加载的模型进行推理或其他操作
        loaded_model = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
        input_tensor_name = \
            loaded_model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs[
                'input'].name
        output_tensor_name = \
            loaded_model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
                'output'].name



        pred = sess.run(output_tensor_name, feed_dict={input_tensor_name: data})
        pred = np.ravel(pred)


    # 反归一化
    pred = pred * (Label_MAX - Label_MIN) + Label_MIN


    sess.close()

    """======================计算指标并打印在控制台======================"""

    error_values = error_total(pred, label)
    error_values = np.array(error_values)
    ME = np.mean(error_values)
    STD = np.std(error_values)
    MAE, In_5, In_10, In_15 = MAE_Fun(pred, label)

    print("-------->>")
    print("平均误差：" + str(ME))
    print("方差：" + str(STD))
    print("平均绝对误差：" + str(MAE))
    print("5mmHg：" + str(In_5))
    print("10mmHg：" + str(In_10))
    print("15mmHg：" + str(In_15))



if __name__ == '__main__':
    export_path = "C:\\Users\Faith\Desktop\血压\模型文件\收缩压\BP-SBP-Saver"
    feaList = []
    # feaList = [2, 34, 1, 9, 3, 5, 12, 16, 20, 21, 25, 6, 37, 22, 15, 19, 32, 4, 11, 10, 33, 27, 35, 30, 36, 38, 13, 28, 39, 14, 29, 18, 24]


    f_train = open("remoteTrainData.csv", 'r', encoding='UTF-8')
    train_datalael = pd.read_csv(f_train, header=None, error_bad_lines=False)
    train_datalael = train_datalael.iloc[:, 1:]
    train_datalael = train_datalael.values
    train_datalael = train_datalael.astype(np.float)

    if (len(feaList) == 0):
        train_data = train_datalael[:, :-2]
    else:
        train_data = train_datalael[:, feaList]
    train_label = train_datalael[:, -2]

    f_valid = open("remoteTestData.csv", 'r', encoding='UTF-8')
    valid_datalael = pd.read_csv(f_valid, header=None, error_bad_lines=False)
    valid_datalael = valid_datalael.iloc[:, 1:]
    valid_datalael = valid_datalael.values
    valid_datalael = valid_datalael.astype(np.float)


    if (len(feaList) == 0):
        valid_data = valid_datalael[:, :-2]
    else:
        valid_data = valid_datalael[:, feaList]
    valid_label = valid_datalael[:, -2]


    """===========数据归一化==========="""
    # 对数据进行归一化
    mean_x = np.mean(train_data, axis=0)
    std_x = np.std(train_data, axis=0)

    train_data = (train_data - mean_x) / std_x
    valid_data = (valid_data - mean_x) / std_x

    # 标签归一化
    Label_MAX = np.max(train_label)
    Label_MIN = np.min(train_label)
    train_label = (train_label - Label_MIN) / (Label_MAX - Label_MIN)


    """BP模型参数"""
    num_classes = 1
    input_size = len(train_data[0])
    hidden_units_size = 2 * input_size + 1
    batch_size = 64
    training_iterations = 1000
    learning_rate = 0.01


    modelValid(valid_data,valid_label)