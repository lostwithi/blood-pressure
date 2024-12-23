import cv2
import numpy as np


def Cg_Compute(bgr_image):
    '''
    :param bgr_image: 传入感兴趣的bgr图，opencv直接读取的帧图片就是bgr格式
    :return: 返回当前帧Cg均值，绿色的浓度偏移量成分
    参考论文：基于人脸视频的心率参数提取，光学精密工程，2020.3
    '''

    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)


    y = ycrcb_image[:,:,0]
    y = y * (y < 230 ) * (y > 60) # y的阈值在60-230之间，不在此范围的为0，在此范围的为1
    y[y > 0] = 1

    cr = ycrcb_image[:,:,1]
    cr = cr * (cr < 180 ) * (cr > 130) # Cr的阈值在130-180之间，不在此范围的为0，在此范围的为1
    cr[cr > 0] = 1

    cb = ycrcb_image[:,:,2]
    cb = cb * (cb < 130 ) * (cb > 72)
    cb[cb > 0] = 1 # Cb的阈值在72-130之间，不在此范围的为0，在此范围的为1

    b = bgr_image[:,:,0] * y * cr * cb
    g = bgr_image[:,:,1] * y * cr * cb
    r = bgr_image[:,:,2] * y * cr * cb

    a_Cg = 128 + (-81.085 * r + 112.0 * g - 30.915 * b)
    # a_Cg = np.mean(a_Cg)
    # a_Cg = np.sum(a_Cg) / (np.sum(a_Cg != 128) )
    a_Cg = (np.sum(a_Cg) - np.sum(a_Cg == 128) * 128) / (np.sum(a_Cg != 128) )

    # cg = []
    #
    #
    #
    # for i in range(len(ycrcb_image)):
    #     for j in range(len(ycrcb_image[0])):
    #         y = ycrcb_image[i][j][0]
    #         cr = ycrcb_image[i][j][1]
    #         cb = ycrcb_image[i][j][2]
    #
    #         b = bgr_image[i][j][0]
    #         g = bgr_image[i][j][1]
    #         r = bgr_image[i][j][2]
    #
    #         if (60 < y < 230 and 72< cb < 130 and 130 < cr < 180):
    #             cg_this = 128 + (-81.085 * r + 112 * g - 30.915 * b)
    #             cg.append(cg_this)
    # cg = np.array(cg)
    # cg_mean = np.mean(cg)
    # print()

    return a_Cg


def LUV_Compute(bgr_image):

    luv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Luv)

    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)



    y = ycrcb_image[:,:,0]
    y = y * (y < 230 ) * (y > 60) # y的阈值在60-230之间，不在此范围的为0，在此范围的为1
    y[y > 0] = 1

    cr = ycrcb_image[:,:,1]
    cr = cr * (cr < 180 ) * (cr > 130) # Cr的阈值在130-180之间，不在此范围的为0，在此范围的为1
    cr[cr > 0] = 1

    cb = ycrcb_image[:,:,2]
    cb = cb * (cb < 130 ) * (cb > 72)
    cb[cb > 0] = 1 # Cb的阈值在72-130之间，不在此范围的为0，在此范围的为1


    u = luv_image[:,:,1]  * y * cr * cb


    return np.mean(u)


def Lab_Space(bgr_image):

    # ycrcb颜色转换用于检测肤色
    ycrcb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    # 转换到Lab颜色空间用于信号提取
    Lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab )


    # 得到肤色阈值numpy数组
    y = ycrcb_image[:,:,0]
    y = y * (y < 230 ) * (y > 60) # y的阈值在60-230之间，不在此范围的为0，在此范围的为1
    y[y > 0] = 1

    cr = ycrcb_image[:,:,1]
    cr = cr * (cr < 180 ) * (cr > 130) # Cr的阈值在130-180之间，不在此范围的为0，在此范围的为1
    cr[cr > 0] = 1

    cb = ycrcb_image[:,:,2]
    cb = cb * (cb < 130 ) * (cb > 72)
    cb[cb > 0] = 1 # Cb的阈值在72-130之间，不在此范围的为0，在此范围的为1

    # Lab颜色空间的A通道和B通道用于信号提取
    A = Lab_image[:,:,1]  * y * cr * cb

    B = Lab_image[:,:,2]  * y * cr * cb

    return np.mean(A),np.mean(B)