import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matlab

def PWA(data,valleys,eng):

    return_Feature = []
    feature = []

    # # 时间特征
    # SUT = 0  # 收缩压上升时间差
    # DT = 0  # 舒张压下降时间差
    # SW_10 = 0
    # SW_25 = 0
    # SW_33 = 0
    # SW_50 = 0
    # SW_75 = 0
    # DW_10 = 0
    # DW_25 = 0
    # DW_33 = 0
    # DW_50 = 0
    # DW_75 = 0
    #
    # # 斜率特征
    # up_slope = 0
    # down_slope = 0
    #
    # #面积特征
    # up_area = 0
    # down_area = 0
    #
    # # 幅值特征
    # SW_10_Height = 0
    # SW_25_Height = 0
    # SW_33_Height = 0
    # SW_50_Height = 0
    # SW_75_Height = 0
    #
    #
    # DW_10_Height = 0
    # DW_25_Height = 0
    # DW_33_Height = 0
    # DW_50_Height = 0
    # DW_75_Height = 0
    #
    # peak_to_start = 0 # 峰值与收缩起点幅值差
    # peak_to_end = 0 # 峰值与舒张压终点幅值差
    #
    # # 宽度特征
    # width_1 = 0
    # width_2 = 0
    # width_3 = 0
    # width_4 = 0
    # width_5 = 0
    # width_6 = 0
    # width_7= 0
    # width_8 = 0
    # width_9= 0
    # width_10 = 0
    for i in range(len(valleys) - 1):
        cur_seg = data[valleys[i]:valleys[i + 1]]

        # cur_seg = data[valleys[0]:valleys[1]]
        # 时间特征
        time = time_feature(cur_seg)
        # 斜率特征
        slope = Slope_feature(cur_seg)

        # 面积特征
        area = Area_feature(cur_seg)

        feature = []
        feature.extend(time)
        feature.extend(area)
        feature.extend(slope)

        """获取高斯参数"""
        matlabForm = matlab.double(cur_seg.tolist())
        gaussianParams = eng.getGaussianParams(matlabForm)
        gaussianParams = np.array(gaussianParams)
        gaussianParams = np.ravel(gaussianParams)
        gaussianParams = gaussianParams.tolist()
        feature.extend(gaussianParams)

        # 幅值特征
        # height = Height_feature(cur_seg)

        # # 宽度特征
        # width = width_feature(cur_seg)




        if (i == 0):
            return_Feature = np.zeros(len(feature),dtype=float)
        return_Feature = return_Feature + feature

    return_Feature = return_Feature / (len(valleys) - 1)
    return_Feature = return_Feature.tolist()

    return return_Feature


#
# def width_feature(cur_seg):
#     cur_seg = mapmaxmin(cur_seg,1,0)
#     cur_peak_value = np.max(cur_seg)
#     cur_peak_point = np.argmax(cur_seg)
#
#     firstHalf = cur_seg[0:cur_peak_point]  # 收缩压上升段
#     secondHalf = cur_seg[cur_peak_point:]
#
#     height_width = np.min([cur_peak_value - np.min(firstHalf), cur_peak_value - np.min(secondHalf)])
#
#     # 宽度特征
#     # width_1 = 0
#     # width_2 = 0
#     # width_3 = 0
#     # width_4 = 0
#     # width_5 = 0
#     # width_6 = 0
#     # width_7 = 0
#     # width_8 = 0
#     # width_9 = 0
#     # width_10 = 0
#
#
#     height_SW = cur_peak_value - np.min(firstHalf)
#     height_DW = cur_peak_value - np.min(secondHalf)
#
#     height_1 = height_width * (1 / 11)
#     height_2 = height_width * (2 / 11)
#     height_3 = height_width * (3 / 11)
#     height_4 = height_width * (4 / 11)
#     height_5 = height_width * (5 / 11)
#     height_6 = height_width * (6 / 11)
#     height_7 = height_width * (7 / 11)
#     height_8 = height_width * (8 / 11)
#     height_9 = height_width * (9 / 11)
#     height_10 = height_width * (10 / 11)
#
#
#
#     firstIndex_1 = np.argmin(abs(np.array(firstHalf) - height_1))
#     secondIndex_1 = np.argmin(abs(np.array(secondHalf) - height_1))
#     width_1 = cur_peak_point - firstIndex_1 + secondIndex_1
#
#     firstIndex_2 = np.argmin(abs(np.array(firstHalf) - height_2))
#     secondIndex_2 = np.argmin(abs(np.array(secondHalf) - height_2))
#     width_2 = cur_peak_point - firstIndex_2 + secondIndex_2
#
#     firstIndex_3 = np.argmin(abs(np.array(firstHalf) - height_3))
#     secondIndex_3 = np.argmin(abs(np.array(secondHalf) - height_3))
#     width_3 = cur_peak_point - firstIndex_3 + secondIndex_3
#
#     firstIndex_4 = np.argmin(abs(np.array(firstHalf) - height_4))
#     secondIndex_4 = np.argmin(abs(np.array(secondHalf) - height_4))
#     width_4 = cur_peak_point - firstIndex_4 + secondIndex_4
#
#     firstIndex_5 = np.argmin(abs(np.array(firstHalf) - height_5))
#     secondIndex_5 = np.argmin(abs(np.array(secondHalf) - height_5))
#     width_5 = cur_peak_point - firstIndex_5 + secondIndex_5
#
#     firstIndex_6 = np.argmin(abs(np.array(firstHalf) - height_6))
#     secondIndex_6 = np.argmin(abs(np.array(secondHalf) - height_6))
#     width_6 = cur_peak_point - firstIndex_6 + secondIndex_6
#
#     firstIndex_7 = np.argmin(abs(np.array(firstHalf) - height_7))
#     secondIndex_7 = np.argmin(abs(np.array(secondHalf) - height_7))
#     width_7 = cur_peak_point - firstIndex_7 + secondIndex_7
#
#     firstIndex_8 = np.argmin(abs(np.array(firstHalf) - height_8))
#     secondIndex_8 = np.argmin(abs(np.array(secondHalf) - height_8))
#     width_8 = cur_peak_point - firstIndex_8 + secondIndex_8
#
#     firstIndex_9 = np.argmin(abs(np.array(firstHalf) - height_9))
#     secondIndex_9 = np.argmin(abs(np.array(secondHalf) - height_9))
#     width_9 = cur_peak_point - firstIndex_9+ secondIndex_9
#
#     firstIndex_10 = np.argmin(abs(np.array(firstHalf) - height_10))
#     secondIndex_10 = np.argmin(abs(np.array(secondHalf) - height_10))
#     width_10 = cur_peak_point - firstIndex_10 + secondIndex_10
#
#     feature = []
#     feature.extend([width_1,width_2,width_3,width_4,width_5,width_6,width_7,width_8,width_9,width_10])
#
#     return feature



def time_feature(cur_seg):

    cur_seg = mapmaxmin(cur_seg,1,0)
    cur_peak_value = np.max(cur_seg)
    cur_peak_point = np.argmax(cur_seg)

    firstHalf = cur_seg[0:cur_peak_point]  # 收缩压上升段
    secondHalf = cur_seg[cur_peak_point:]


    SUT = cur_peak_point
    DT = len(cur_seg) - cur_peak_point

    SW_height = cur_peak_value - np.min(firstHalf)
    SW_height_10 = (0.10 * SW_height) + np.min(firstHalf)
    SW_height_25 = (0.25 * SW_height) + np.min(firstHalf)
    SW_height_33 = (0.33 * SW_height) + np.min(firstHalf)
    SW_height_50 = (0.50 * SW_height) + np.min(firstHalf)
    SW_height_75 = (0.75 * SW_height) + np.min(firstHalf)

    DW_height = cur_peak_value - np.min(secondHalf)
    DW_height_10 = (0.10 * DW_height) + np.min(secondHalf)
    DW_height_25 = (0.25 * DW_height) + np.min(secondHalf)
    DW_height_33 = (0.33 * DW_height) + np.min(secondHalf)
    DW_height_50 = (0.50 * DW_height) + np.min(secondHalf)
    DW_height_75 = (0.75 * DW_height) + np.min(secondHalf)


    firstIndex_10 = np.argmin(abs(np.array(firstHalf) - SW_height_10))
    secondIndex_10 = np.argmin(abs(np.array(secondHalf) - DW_height_10)) + cur_peak_point

    firstIndex_25 = np.argmin(abs(np.array(firstHalf) - SW_height_25))
    secondIndex_25 = np.argmin(abs(np.array(secondHalf) - DW_height_25)) + cur_peak_point

    firstIndex_33 = np.argmin(abs(np.array(firstHalf) - SW_height_33))
    secondIndex_33 = np.argmin(abs(np.array(secondHalf) - DW_height_33)) + cur_peak_point

    firstIndex_50 = np.argmin(abs(np.array(firstHalf) - SW_height_50))
    secondIndex_50 = np.argmin(abs(np.array(secondHalf) - DW_height_50)) + cur_peak_point

    firstIndex_75 = np.argmin(abs(np.array(firstHalf) - SW_height_75))
    secondIndex_75 = np.argmin(abs(np.array(secondHalf) - DW_height_75)) + cur_peak_point

    SW_10 = cur_peak_point - firstIndex_10
    DW_10 = secondIndex_10 - cur_peak_point

    SW_25 = cur_peak_point - firstIndex_25
    DW_25 = secondIndex_25 - cur_peak_point

    SW_33 = cur_peak_point - firstIndex_33
    DW_33 = secondIndex_33 - cur_peak_point

    SW_50 = cur_peak_point - firstIndex_50
    DW_50 = secondIndex_50 - cur_peak_point

    SW_75 = cur_peak_point - firstIndex_75
    DW_75 = secondIndex_75 - cur_peak_point

    feature = []
    feature.extend([SUT, DT, SW_10, SW_25, SW_33, SW_50, SW_75, DW_10, DW_25, DW_33, DW_50, DW_75, SUT + DT, SUT - DT, SUT / DT, SUT / (SUT + DT), DT / (SUT + DT)])

    return feature


# def Height_feature(cur_seg):
#
#     cur_peak_value = np.max(cur_seg)
#     cur_peak_point = np.argmax(cur_seg)
#
#     firstHalf = cur_seg[0:cur_peak_point]  # 收缩压上升段
#
#         # 收缩压期间的幅值特征
#     peak_to_start = cur_peak_value - np.min(firstHalf) # 最高点与收缩压起点的幅值差
#     SW_10_Height = peak_to_start * 0.1 + np.min(firstHalf)
#     SW_25_Height = peak_to_start * 0.25 + np.min(firstHalf)
#     SW_33_Height = peak_to_start * 0.33 + np.min(firstHalf)
#     SW_50_Height = peak_to_start * 0.5 + np.min(firstHalf)
#     SW_75_Height = peak_to_start * 0.75 + np.min(firstHalf)
#
#
#     # secondHalf = cur_seg[cur_peak_point:]  # 舒张压下降段
#     # peak_to_end = cur_peak_value - np.min(secondHalf)
#     # DW_10_Height = peak_to_end * 0.1 + np.min(secondHalf)
#     # DW_25_Height = peak_to_end * 0.25 + np.min(secondHalf)
#     # DW_33_Height = peak_to_end * 0.33 + np.min(secondHalf)
#     # DW_50_Height = peak_to_end * 0.5 + np.min(secondHalf)
#     # DW_75_Height = peak_to_end * 0.75 + np.min(secondHalf)
#
#     feature = []
#     feature.extend([cur_peak_value,cur_seg[0],peak_to_start,SW_10_Height,SW_25_Height,SW_33_Height,SW_50_Height,SW_75_Height])
#
#
#     return feature


def Area_feature(cur_seg):

    cur_peak_value = np.max(cur_seg)
    cur_peak_point = np.argmax(cur_seg)

    up_area = 0
    down_area = 0


    # 计算面积 *--->等同于逐点相加
    for point in range(len(cur_seg)):

        # 收缩面积
        if (point < cur_peak_point):
            up_area += np.abs(cur_seg[point])

        # 舒张面积
        if (point > cur_peak_point):
            down_area += np.abs(cur_seg[point])

    totalArea = up_area + down_area

    Pm = totalArea / len(cur_seg)


    K = (Pm - np.min(cur_seg)) / (np.max(cur_seg) - np.min(cur_seg))



    ratio = float(up_area / down_area)

    feature = []
    feature.extend([up_area,down_area, up_area + down_area, ratio, up_area / (up_area + down_area), down_area / (up_area + down_area),cur_peak_value, cur_seg[0], K])


    return feature


def Slope_feature(cur_seg):
    cur_peak_value = np.max(cur_seg)
    cur_peak_point = np.argmax(cur_seg)

    # 计算斜率
    up_slope =  (cur_peak_value - cur_seg[0]) / (cur_peak_point)
    down_slope = (cur_seg[-1]- cur_peak_value) / (len(cur_seg) - cur_peak_point)

    feature = []
    feature.extend([up_slope,down_slope])
    return feature

def mapmaxmin(data,ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax-ymin)*(data-xmin) / (xmax - xmin) + ymin
    return new_data