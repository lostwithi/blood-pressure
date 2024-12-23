import time

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math



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



"""=======================训练模型==================="""
def train(train_data,train_label):
    tf.reset_default_graph()


    """=======================输入权重并做分析==============================="""

    X = tf.placeholder(tf.float32, shape=[None, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    W1 = tf.Variable(tf.random_normal([input_size, hidden_units_size], stddev=0.1))
    B1 = tf.Variable(tf.random_normal([1, hidden_units_size], stddev=0.1))
    W2 = tf.Variable(tf.random_normal([hidden_units_size, num_classes], stddev=0.1))
    B2 = tf.Variable(tf.random_normal([1, num_classes], stddev=0.1))

    """读取权重文件"""
    """==================================读取权重==========================="""

    weights_file = open("SSA-MIV-SBPTargetWeight.csv",  'r',
                        encoding='UTF-8')
    weights = pd.read_csv(weights_file, header=None, error_bad_lines=False)
    weights = weights.values
    x = np.ravel(weights)

    W1 = x[0: input_size * hidden_units_size]
    W1 = np.reshape(W1, (input_size, hidden_units_size))
    W1 = tf.Variable(W1, dtype=tf.float32)

    B1 = x[input_size * hidden_units_size: input_size * hidden_units_size + hidden_units_size]
    B1 = np.reshape(B1, (1, hidden_units_size))
    B1 = tf.Variable(B1, dtype=tf.float32)

    W2 = x[
         input_size * hidden_units_size + hidden_units_size: input_size * hidden_units_size + hidden_units_size + hidden_units_size * num_classes]
    W2 = np.reshape(W2, (hidden_units_size, num_classes))
    W2 = tf.Variable(W2, dtype=tf.float32)

    B2 = x[input_size * hidden_units_size + hidden_units_size + hidden_units_size * num_classes:]
    B2 = np.reshape(B2, (1, num_classes))
    B2 = tf.Variable(B2, dtype=tf.float32)

    """开始训练"""

    hidden_opt = tf.matmul(X, W1) + B1
    hidden_opt = tf.nn.relu(hidden_opt)
    final_opt = tf.matmul(hidden_opt, W2) + B2
    final_opt = tf.nn.relu(final_opt)

    loss = tf.losses.mean_squared_error(predictions=final_opt, labels=Y)
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    train_loss = []


    with tf.Session() as sess:
        sess.run(init)

        """保存权重系数"""
        weight = []
        weight.extend(np.ravel(W1.eval(sess)).tolist())
        weight.extend(np.ravel(B1.eval(sess)).tolist())
        weight.extend(np.ravel(W2.eval(sess)).tolist())
        weight.extend(np.ravel(B2.eval(sess)).tolist())
        weight = pd.DataFrame(weight)
        weight.to_csv("weight.csv", index = False, header = False)


        datasize = len(train_data)
        total_batch = int(math.ceil(datasize / batch_size))

        for i in range(training_iterations):

            avg_loss = 0

            for j in range(total_batch):
                start = (j * batch_size)
                end = np.min([start + batch_size, datasize])
                x = train_data[start:end]
                y = train_label[start:end]
                y = y.reshape(-1, num_classes)
                _, cost = sess.run([opt, loss],
                                   feed_dict={X: x, Y: y})
                avg_loss += cost / total_batch

            if (i % 200 == 0):
                print("第" + str(i) + ":", avg_loss)


        # 创建保存模型的目录
        export_path = "./SSA-MIV-BP-SBP-Saver"
        # 保存训练模型
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input': tf.saved_model.utils.build_tensor_info(X)},
            outputs={'output': tf.saved_model.utils.build_tensor_info(final_opt)},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()



    sess.close()




