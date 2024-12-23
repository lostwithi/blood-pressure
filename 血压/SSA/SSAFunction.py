import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn.model_selection import KFold


class function:
    def __init__(self,input_size,hidden_units_size,num_classes,inputData,inputLabel,
                 batch_size,training_iterations,learning_rate,pop):
        print("function init")


        self.input_size = input_size
        self.hidden_units_size = hidden_units_size
        self.num_classes = num_classes


        self.inputData = inputData
        self.inputLabel = inputLabel


        self.batch_size = batch_size
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        self.pop = pop

    def Parameters(self,F):


        if F=='F1':
            ParaValue=[-100,100,30]

        return ParaValue

    def fun(self,F,x):

        X_var = x.copy()
        input_size = self.input_size
        hidden_units_size = self.hidden_units_size
        num_classes = self.num_classes

        trainFeature = self.trainFeature
        trainLabel = self.trainLabel
        testFeature = self.testFeature
        testLabel = self.testLabel



        batch_size = self.batch_size
        training_iterations = self.training_iterations
        learning_rate = self.learning_rate

        """数据归一化"""

        # 特征归一化
        mean_x = np.mean(trainFeature, axis=0)
        std_x = np.std(trainFeature, axis=0)
        trainFeature = (trainFeature - mean_x) / std_x
        testFeature = (testFeature - mean_x) / std_x

        # 标签归一化
        Label_MAX = np.max(trainLabel)
        Label_MIN = np.min(trainLabel)

        trainLabel = (trainLabel - Label_MIN) / (Label_MAX - Label_MIN)




        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, shape=[None, input_size])
        Y = tf.placeholder(tf.float32, shape=[None, num_classes])

        W1 = x[0: input_size * hidden_units_size]
        W1 = np.reshape(W1, (input_size, hidden_units_size))
        W1 = tf.Variable(W1, dtype=tf.float32)

        B1 = x[input_size * hidden_units_size: input_size * hidden_units_size + hidden_units_size]
        B1 = np.reshape(B1, (1, hidden_units_size))
        B1 = tf.Variable(B1, dtype=tf.float32)
        # B1 = tf.reshape(B1, [hidden_units_size])

        W2 = x[
             input_size * hidden_units_size + hidden_units_size: input_size * hidden_units_size + hidden_units_size + hidden_units_size * num_classes]
        W2 = np.reshape(W2, (hidden_units_size, num_classes))
        W2 = tf.Variable(W2, dtype=tf.float32)

        B2 = x[input_size * hidden_units_size + hidden_units_size + hidden_units_size * num_classes:]
        B2 = np.reshape(B2, (1, num_classes))
        B2 = tf.Variable(B2, dtype=tf.float32)

        # 定义BP网络层
        hidden_opt = tf.matmul(X, W1) + B1
        hidden_opt = tf.nn.relu(hidden_opt)
        final_opt = tf.matmul(hidden_opt, W2) + B2
        final_opt = tf.nn.relu(final_opt)

        # 定义损失函数、优化器以及初始方法
        loss = tf.losses.mean_squared_error(predictions=final_opt, labels=Y)
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()


        """交叉验证寻优参数值"""
        errorsValue = []

        kf = KFold(n_splits=5, shuffle=False)  # 初始化KFold
        for train_index, test_index in kf.split(self.inputData):  # 调用split方法切分数据
            # print('train_index:%s , test_index: %s ' % (train_index, test_index))
            trainFeature = self.inputData[train_index, :]

            trainLabel = self.inputLabel[train_index]

            testFeature = self.inputData[test_index, :]
            testLabel = self.inputLabel[test_index]
            """数据归一化"""

            # 特征归一化
            mean_x = np.mean(trainFeature, axis=0)
            std_x = np.std(trainFeature, axis=0)
            trainFeature = (trainFeature - mean_x) / std_x
            testFeature = (testFeature - mean_x) / std_x

            # max_x = np.max(trainFeature, axis=0)
            # min_x = np.min(trainFeature, axis=0)
            # trainFeature = (trainFeature - min_x) / (max_x - min_x)
            # testFeature = (testFeature - min_x) / (max_x - min_x)

            # 标签归一化
            Label_MAX = np.max(trainLabel)
            Label_MIN = np.min(trainLabel)

            trainLabel = (trainLabel - Label_MIN) / (Label_MAX - Label_MIN)

            with tf.Session() as sess:
                sess.run(init)

                # print("初始权重:", sess.run(B1))
                datasize = len(trainFeature)
                total_batch = int(math.ceil(datasize / batch_size))

                for i in range(training_iterations):

                    avg_loss = 0

                    for j in range(total_batch):
                        start = (j * batch_size)
                        end = np.min([start + batch_size, datasize])
                        x = trainFeature[start:end]
                        y = trainLabel[start:end]

                        y = y.reshape(-1, num_classes)

                        _, cost = sess.run([opt, loss],
                                           feed_dict={X: x, Y: y})
                        avg_loss += cost / total_batch


                test_pred = sess.run(final_opt, feed_dict={X: testFeature})
                test_pred = np.ravel(test_pred)

            # 反归一化
            test_pred = test_pred * (Label_MAX - Label_MIN) + Label_MIN
            # 计算损失值

            errors = testLabel - test_pred
            errorsValue.extend(errors)


        errorsValue = np.abs(errorsValue)
        MAE = np.sum(errorsValue) / len(errorsValue)
        print(MAE)

        sess.close()


        res = MAE

        return res


    def Bounds(self,s,Lb,Ub):
        temp=s
        for i in range(len(s)):
            if temp[i]<Lb[0,i]:
                temp[i]=Lb[0,i]
            elif temp[i]>Ub[0,i]:
                temp[i]=Ub[0,i]

        return temp


    def SSA(self,pop,M,c,d,dim,f):
        """
        :param pop: 种群数量
        :param M: 最大迭代次数
        :param c: 下界
        :param d: 上界
        :param dim: 问题维度
        :param f: 问题函数
        :return: 返回解
        """
        #global fit
        P_percent=0.2
        pNum=round(pop*P_percent)
        lb=c*np.ones((1,dim))
        ub=d*np.ones((1,dim))
        # X=np.zeros((pop,dim))
        X = self.pop
        fit=np.zeros((pop,1))

        print("==================计算初始种群适应度==================")
        for i in range(pop):
            # X[i,:]=lb+(ub-lb)*np.random.rand(1,dim)
            fit[i,0]=self.fun(f,X[i,:])
        print("==================计算初始种群适应度==================")
        pFit=fit.copy()
        pX=X.copy()
        fMin=np.min(fit[:,0])
        bestI=np.argmin(fit[:,0])
        bestX=X[bestI,:].copy()
        Convergence_curve=np.zeros((1,M))


        for t in range(M):

            print("第几轮==========",t)

            sortIndex=np.argsort(pFit.T)
            fmax=np.max(pFit[:,0])
            B=np.argmax(pFit[:,0])
            worse=X[B,:]


            """================探索者的位置更新========================="""
            r2=np.random.rand(1)
            if r2 < 0.8: # 预警值较小，说明没有捕食者出现
                for i in range(pNum):


                    r1=np.random.rand(1)
                    X[sortIndex[0,i],:]=pX[sortIndex[0,i],:]*np.exp(-(i)/(r1*M)) # 对变量做随机变换
                    X[sortIndex[0,i],:]=self.Bounds(X[sortIndex[0,i],:],lb,ub) # 规范边界
                    fit[sortIndex[0,i],0]=self.fun(f,X[sortIndex[0,i],:]) # 计算新的适应度

                    print("==============小于预警值的探索者===========================")

            elif r2 >= 0.8: # 预警值较大，说明又捕食者威胁到种群安全，需要到其他地方觅食
                for i in range(pNum):

                    X[sortIndex[0,i],:]=pX[sortIndex[0,i],:]+np.random.rand(1)*np.ones((1,dim))
                    X[sortIndex[0,i],:]=self.Bounds(X[sortIndex[0,i],:],lb,ub)
                    fit[sortIndex[0,i],0]=self.fun(f,X[sortIndex[0,i],:])

                    print("==============大于预警值的探索者===========================")


            bestII=np.argmin(fit[:,0])
            bestXX=X[bestII,:]

            """================追随者的位置更新========================="""

            for ii in range(pop-pNum):


                i=ii+pNum
                A=np.floor(np.random.rand(1,dim)*2)*2-1
                if i> pop/2:
                    X[sortIndex[0,i],:]=np.random.rand(1)*np.exp(worse-pX[sortIndex[0,i],:]/np.square(i))
                else:
                    X[sortIndex[0,i],:]=bestXX+np.dot(np.abs(pX[sortIndex[0,i],:]-bestXX),1/(A.T*np.dot(A,A.T)))*np.ones((1,dim))
                X[sortIndex[0,i],:]=self.Bounds(X[sortIndex[0,i],:],lb,ub)
                fit[sortIndex[0,i],0]=self.fun(f,X[sortIndex[0,i],:])

                print("==============追随者===========================")


            arrc = np.arange(len(sortIndex[0,:]))
            #c=np.random.shuffle(arrc)
            c=np.random.permutation(arrc)
            b=sortIndex[0,c[0:20]]
            for j in range(len(b)):
                if pFit[sortIndex[0,b[j]],0]>fMin:
                    X[sortIndex[0,b[j]],:]=bestX+np.random.rand(1,dim)*np.abs(pX[sortIndex[0,b[j]],:]-bestX)
                else:
                    X[sortIndex[0,b[j]],:]=pX[sortIndex[0,b[j]],:]+(2*np.random.rand(1)-1)*np.abs(pX[sortIndex[0,b[j]],:]-worse)/(pFit[sortIndex[0,b[j]]]-fmax+10**(-50))
                X[sortIndex[0,b[j]],:]=self.Bounds(X[sortIndex[0,b[j]],:],lb,ub)
                fit[sortIndex[0,b[j]],0]=self.fun(f,X[sortIndex[0,b[j]]])


            for i in range(pop):

                if fit[i,0]<pFit[i,0]:
                    pFit[i,0]=fit[i,0]
                    pX[i,:]=X[i,:]
                if pFit[i,0]<fMin:
                    fMin=pFit[i,0].copy()
                    bestX=pX[i,:].copy()
                    print(fMin)
                    print(bestX)
            Convergence_curve[0,t]=fMin


            record = pd.DataFrame(pFit.T)
            record.to_csv("适应更新.csv",index=False,header=False,mode="a")

        return fMin,bestX,Convergence_curve






