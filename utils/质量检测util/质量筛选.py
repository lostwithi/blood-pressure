import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import detrend, butter, filtfilt
from scipy.fftpack import fft
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


# plot_switch = True
plot_switch = False
low_prominence = 0.2

def mapmaxmin(data, ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax - ymin) * (data - xmin) / (xmax - xmin) + ymin
    return new_data

#信号处理方法
def chrom(data_r, data_g, data_b):
    # 数据的融合还有待商量
    x_s = 3 * data_r - 2 * data_g
    y_s = 1.5 * data_r + data_g - 1.5 * data_b
    alpha = np.std(x_s) / np.std(y_s)
    return 3 * (1 - alpha / 2) * data_r - 2 * (1 + alpha / 2) * data_g + 3 * alpha * data_b / 2


def inter_array(orin_arr, FPS):
    # print('np.shape(orin_arr)', np.shape(orin_arr))
    x_oringin = np.linspace(0, 15, len(orin_arr))
    interp1d_f = interp1d(x_oringin, orin_arr, kind='cubic')
    xnew = np.linspace(0, 15, num=int(15 * FPS))  # fps:摄像头帧率
    data_new_inter = interp1d_f(xnew)
    return data_new_inter


def get_real_count_value(data, length_number, distance, prominence):
    try:
        single_peaks_hr, _ = find_peaks(data, distance=distance, prominence=[prominence, None])
        start_index = single_peaks_hr[0]
        end_index = single_peaks_hr[-1]
        hr_value = length_number/(end_index - start_index) * (len(single_peaks_hr) - 1)
    except:
        hr_value = 0
    return hr_value


def get_real_fft_value(data, fps):
    try:
        fft_data = abs(fft(data))  # 归一化处理
        start_index = int((len(fft_data)) * 0.5 / fps)
        end_index = int((len(fft_data)) * 4 / fps)

        single_fft_data = np.asarray(fft_data[start_index:end_index])
        maxindex = np.argmax(single_fft_data) + start_index
        hr_value_fft = maxindex / len(data) * fps * 60
    except:
        hr_value_fft = 0
    return hr_value_fft

def sliding_window_for_optimal_segment_index(data, window_step, window_length, fps):
    def judge_signal_usable(data, fps):
        def signal_reconsitution(data):
            """
            #去除信号的超低频噪声，完成信号的基本波形校正(去除基线漂移)
            :param data: 原始信号
            :return: 去除低频之后的信号
            """
            data_green_raw = np.array(data)
            # 将candidate_data映射到[-1, 1]区间
            candidate_data = mapmaxmin(data_green_raw, 1, -1)
            # 获取超低频数据
            b, a = signal.butter(3, 0.02, 'low')
            motion_arti = signal.filtfilt(b, a, candidate_data)
            # 获取去除基线漂移后的数据
            candidate_data = candidate_data - motion_arti
            return candidate_data

        def deal_by_moving_average(data):
            def moving_average(interval, windowsize):
                """
                :param interval: 输入的原始数据
                :param windowsize:  均值的窗宽大小
                :return: 均值滑动之后的一维数据
                """
                window = np.ones(int(windowsize)) / float(windowsize)
                re = np.convolve(interval, window, 'same')
                return re

            choosed_data = moving_average(data, 4)
            return mapmaxmin(choosed_data, 1, -1)

        def box_plot_outliers(deg, tolerance, name):
            if isinstance(deg, list):
                import numpy as np
                deg = np.asarray(deg)
            import matplotlib.pyplot as plt
            import numpy as np
            # 生成一些示例数据
            data = deg

            # # 创建箱型图
            # fig, ax = plt.subplots()
            # ax.boxplot(data)
            #
            # percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
            # # 标注离群值
            # q1 = percentile[0]  # 上四分位数
            # q3 = percentile[2]  # 下四分位数
            #
            # iqr = q3 - q1
            # lower_bound = q1 - 1.5 * iqr
            # upper_bound = q3 + 1.5 * iqr
            # outliers = data[(data < lower_bound) | (data > upper_bound)]
            # ax.scatter(np.ones(len(outliers)), outliers, marker='o', color='red', alpha=0.5)
            # # 显示图形
            # plt.show()

            """
            deg: 传入的需要判断的数据
            tolerance: 容忍度，即为允许出现多少个异常值
            name: 当前判断数据的名字
            描述：对传入的数据进行箱型图的异常值判断，根据tolerance程序，返回是否可用的布尔函数
            return -> boolean
            """
            mean = np.mean(deg)
            var = np.var(deg)

            percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
            # print("分位数：", percentile)
            # 以下为箱线图的五个特征值
            Q1 = percentile[0]  # 上四分位数
            Q3 = percentile[2]  # 下四分位数
            IQR = Q3 - Q1  # 四分位距
            ulim = Q3 + 1.5 * IQR  # 上限 非异常范围内的最大值
            llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

            new_deg = []
            for i in range(len(deg)):
                if (llim < deg[i] and deg[i] < ulim):
                    new_deg.append(deg[i])
            # print(name + "--箱形图数据差：", len(deg) - len(new_deg), "个。")
            # 根据是否有异常值返回布尔值
            if len(deg) - len(new_deg) <= tolerance:
                return True
            else:
                return False

        def judge_peak_valley_distance_outliers_1(data, fps):
            """
            data: 必须为原始信号处理经过signal_reconsitution与deal_by_moving_average之后的数据
            将data传入，对data进行多维度的质量判断，如果是符合所有维度的判断，那么这一段信号就是可用的
            否者是不可用的
            :param data: 原始数据
            :param fps: 帧率
            :return: Boolean 是否可用的标识
            """
            # 传入的是原始波形
            # 判断当前波形是否含有nan，含有nan则表示数据缺失
            if pd.DataFrame(data).isnull().values.any() == True:
                # print("截取的波段含有nan，原因是采集的数据缺失")
                return False, -1
            data = mapmaxmin(data, 1, -1)
            # 第一个判断条件：此段的数峰值的数量必须大于40
            if get_real_count_value(data=data, length_number=fps * 60, distance=12.5, prominence=0.2) <= 40:
                print("峰值数量有问题")
                return False, -1

            # 第二个判断条件：对这段信号本身的质量进行检测
            # elif box_plot_outliers(data, tolerance=0, name="信号本身的质量") == False:
            #     print("信号本身的质量有问题")
            #     return False, -1

            else:
                peak_indexs, _ = find_peaks(data, distance=12.5, prominence=[0.2, None])
                trough_indexs, _ = find_peaks(-1 * data, distance=12.5, prominence=[0.2, None])
                peaks_values = [data[x] for x in peak_indexs]
                troughs_values = [data[x] for x in trough_indexs]

                # 第3个判断条件：保证峰值索引的下一个必须是谷值索引
                list_peak2valley_height = []
                if peak_indexs[0] < trough_indexs[0]:
                    if len(peak_indexs) < len(trough_indexs):
                        # print("第一数值是峰值，但是峰的数量小于谷的数量")
                        return False, -1
                    else:
                        for cur_index in np.arange(0, len(trough_indexs), 1):
                            if peak_indexs[cur_index] > trough_indexs[cur_index]:
                                # print("峰值索引的下一个不是谷值索引")
                                return False, -1
                            else:
                                list_peak2valley_height.append(
                                    data[peak_indexs[cur_index]] - data[trough_indexs[cur_index]])
                else:
                    if len(peak_indexs) > len(trough_indexs):
                        # print("第一数值是谷值，但是谷的数量小于峰的数量")
                        return False, -1
                    else:
                        for cur_index in np.arange(0, len(peak_indexs), 1):
                            if peak_indexs[cur_index] < trough_indexs[cur_index]:
                                # print("谷值索引的下一个不是峰值索引")
                                return False, -1
                            else:
                                list_peak2valley_height.append(
                                    data[peak_indexs[cur_index]] - data[trough_indexs[cur_index]])

                # 峰值
                if box_plot_outliers(peaks_values, tolerance=0, name="峰谷高度差") == False:
                    print("峰值的异常值")
                    return False, -1

                # 谷值
                if box_plot_outliers(troughs_values, tolerance=0, name="峰谷高度差") == False:
                    print("谷值的异常值")
                    return False, -1

                # 第4个判断条件：峰谷高度差的异常值判断
                if box_plot_outliers(list_peak2valley_height, tolerance=0, name="峰谷高度差") == False:
                    print("峰谷高度差的异常值")
                    return False, -1

                # 第5个判断条件：峰峰值距离异常值判断
                list_peak2peak_distance = []
                for single_peak_index in np.arange(0, len(peak_indexs) - 1, 1):
                    list_peak2peak_distance.append(peak_indexs[single_peak_index + 1] - peak_indexs[single_peak_index])
                if box_plot_outliers(list_peak2peak_distance, tolerance=0, name="峰峰距离") == False:
                    print("两个峰峰距离异常值")
                    return False, -1
                # 第6个判断条件：谷谷值距离异常值判断
                list_valley2valley_distance = []
                for single_trough_index in np.arange(0, len(trough_indexs) - 1, 1):
                    list_valley2valley_distance.append(
                        trough_indexs[single_trough_index + 1] - trough_indexs[single_trough_index])
                if box_plot_outliers(list_valley2valley_distance, tolerance=0, name="谷谷距离") == False:
                    print("两个谷谷值距离异常值判断")
                    return False, -1

                # 第7个判断条件
                # 按照30个点进行一次sd的检验，如果sd过大，或者不在容错的区间，就抛弃
                for single_second_index in np.arange(0, int(len(data) / 30), 1):
                    single_second_data = data[single_second_index * 30: (single_second_index + 1) * 30]
                    # print(np.std(single_second_data))

                # 当所有的判断条件都没有出错的时候，就可以返回True
                return True, 1

        # 低通滤波去除低频噪声
        judge_signal = signal_reconsitution(data)

        # 均值滑动去除高频噪声
        judge_signal_moving_data = deal_by_moving_average(judge_signal)
        # judge_signal_moving_data = np.ravel(data)

        flag, level = judge_peak_valley_distance_outliers_1(judge_signal_moving_data, fps)
        return flag, level

    flag = False
    level = -2
    res_start = None
    res_end = None
    #第一个循环是：优先15s的滑窗，然后若15s没有数据可用，那么就将滑窗-1，变成14s
    while flag == False and window_length >= 5:
        #每一次重新开始滑窗时，index必须是0
        #   然后开始的索引为：(index * window_step * fps)
        #   然后结尾的索引为：(index + window_length) * window_step * fps，相差为一个窗宽
        index = 0
        start_index = index *  fps
        end_index = (index + window_length) *  fps

        #第二个循环是：滑窗按照步长移动，1s移动一次
        # 保证当前end索引与start的索引相差必须超过5s, 并且end_index必须是小于数据长度的
        while end_index - start_index >= fps * window_length and end_index <= len(data):
            #进入这里表示start索引与end索引组成的segment数据为待判断的数据，并且end索引-start索引是>=5s的
            start_index = index * fps
            end_index = (index + window_length) * fps
            #如果end索引的数值超过了data的长度，那么，segment的end索引就改变为len(data)
            if ((index + window_length) * window_step * fps) > len(data):
                end_index = len(data)
            # print("当前的窗宽为：" + str(window_length) + "--" + str(start_index) + ":" + str(end_index))
            if end_index - start_index < fps * window_length:
                break
            data_part = data[start_index:end_index]
            #此时的data_part为当前选取的需要进行判断的segment数据
            flag, level = judge_signal_usable(data_part, fps)
            #判断这一段子信号是否可用，若不可用，则flag==False，
            # 那就将index后移一个，判断下一个滑窗的数据
            # 否则: 即判断此段信号可用，那么就break这个循环且赋值start_index与end_index
            if flag == False:
                index = index + window_step
            else:
                #信号可用：将当前运行的start_index与end_index赋值给res_start与res_end
                res_start = start_index
                res_end = end_index
                flag = True
                break
        #当一个滑窗的长度没有符合的条件的波段时，就开始减小滑窗长度
        if flag == False:
            window_length = window_length - 1
    #根据是否有符合的判断条件进行结果返回
    if flag == False:
        return False, None, None, None
    else:
        # print("可用的start索引与end索引：", res_start, res_end)
        return True, level, res_start, res_end


if __name__ == '__main__':
    root = "D:\zhang\BiopacData\\2020_Data\\totalFile"
    # matlab_eng = matlab.engine.start_matlab()
    # matlab_eng.cd('./ACMD/matlabFile')  # 进入MATLAB引擎加载的m函数所在文件夹





    # root = "D:\chen\论文\IAM\data\DealedWithEVM"
    # for single_fre_band in os.listdir(root):
    #     single_fre_path = os.path.join(root, single_fre_band)
    #     if os.path.isdir(single_fre_path):
    #         for single_type in os.listdir(single_fre_path):
    #             file_name_list = []
    #
    #             file_name = single_type + "_"
    #             single_type_path = os.path.join(single_fre_path, single_type)
    #
    #             if os.path.isdir(single_type_path):
    #                 for single_file in os.listdir(single_type_path):
    #                     single_file_path = os.path.join(single_type_path, single_file)
    #                     print(single_file_path)
    #                     if os.path.isfile(single_file_path):
    #
    #
    #                         single_file_path = "D:\chen\论文\IAM\data\DealedWithEVM\Freq-band-0.5-to-2\亮\\10-亮-2-ideal-from-0.5-to-2-alpha-50-level-6-chromAtn-1.csv"
    #                         # single_file_path = "D:\chen\论文\IAM\data\DealedWithEVM\Freq-band-0.5-to-2\亮\\23-亮-0-ideal-from-0.5-to-2-alpha-50-level-6-chromAtn-1.csv"
    #                         single_file_path = "D:\chen\论文\IAM\data\DealedWithEVM\Freq-band-0.5-to-2\低\\09-低-0-ideal-from-0.5-to-2-alpha-50-level-6-chromAtn-1.csv"
    #
    #
    #                         data = pd.read_csv(single_file_path)

                            # red_data = data["Red"]
                            #
                            # blue_data = data["Blue"]

                            # green_data = data["Green"]

                            # plt.subplot(211)
                            # plt.plot(green_data)
                            #
                            # green_data = chrom(red_data, green_data, blue_data)
                            # plt.subplot(212)
                            # plt.plot(green_data)
                            # plt.show()


                            # file_name_list.append(file_name)
                            # Flag, level, start_index, end_index = sliding_window_for_optimal_segment_index(data=green_data, window_length=15, window_step=1, fps=30)
                            # print(Flag)
                            # print(level)
                            # print(start_index)
                            # print(end_index)
                            # if level == 1:
                            #     print("质量筛选1--成功")
                            #     valid_segment_data_hr = green_data[
                            #                             start_index: end_index]
                            #     # 去除超低频基线漂移
                            #     plt.subplot(211)
                            #     plt.plot(green_data)
                            #     plt.subplot(212)
                            #     plt.plot(valid_segment_data_hr)
                            #     plt.show()





    for file in os.listdir(root):
        # file = "01-低-0.avi.csv"
        print(file)

        file = os.path.join(root, file)
        data = pd.read_csv(file)

        # red_data = data["Red"]
        #
        # blue_data = data["Blue"]

        green_data = data["nose_Green"]
        green_data = mapmaxmin(green_data,1,-1)
        # green_data = green_data[0:30*30]
        Flag, level, start_index, end_index = sliding_window_for_optimal_segment_index(data=green_data, window_length=15, window_step=1, fps=30)
        print(Flag)
        print(level)
        print(start_index)
        print(end_index)
        # if level == 1:
        #     print("质量筛选1--成功")
        #     valid_segment_data_hr = green_data[
        #                             start_index: end_index]
        #     # 去除超低频基线漂移
        #     plt.subplot(211)
        #     plt.plot(green_data)
        #     plt.subplot(212)
        #     plt.plot(valid_segment_data_hr)
        #     plt.show()
        green_data = green_data[: 30 * 30]
        green_data = mapmaxmin(green_data, 1, -1)
        t = np.arange(0, len(green_data)) / 30

        plt.figure(figsize=(15, 4))
        plt.rc('font', family='Times New Roman')
        grid = plt.GridSpec(14, 6, wspace=0.5, hspace=0.4)
        ax1 = plt.subplot(grid[0:12, :])
        plt.plot(t, green_data, color="black", linewidth=1.5)
        plt.xlabel("Time(s)", fontsize=20)
        plt.ylabel("Amplitude", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=16)
        plt.show()
        print()