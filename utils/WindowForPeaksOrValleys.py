import numpy as np

def findValleys(signals,fs):
    valleys = [0]
    width = int(fs * 0.7)

    #滑动窗口寻找峰值或者谷值
    for i in range(len(signals) - width * 2):
        try:
            subarr = signals[i:width + i] #获取当前窗口的信号序列

            middle = int(i + width / 2)  #或许当前窗口信号序列中间的索引值

            if (signals[middle] == min(subarr) and min(subarr) != max(subarr)):

                # 如果谷值点之后的采样点幅值也等于谷值，则排除这个点
                if (middle - valleys[-1] < width / 2):
                    continue

                valleys.append(middle)

        except ValueError:
            break


    return valleys[1:]

def findPeaks(signals,fs):
    peaks = [0]
    width = int(fs * 0.7)

    #滑动窗口寻找峰值或者谷值
    for i in range(len(signals) - width * 2):
        try:
            subarr = signals[i:width + i] #获取当前窗口的信号序列

            middle = int(i + width / 2)  #或许当前窗口信号序列中间的索引值

            if (signals[middle] == max(subarr) and min(subarr) != max(subarr)):

                # 如果谷值点之后的采样点幅值也等于谷值，则排除这个点
                if (middle - peaks[-1] < width / 2):
                    continue

                peaks.append(middle)

        except ValueError:
            break


    return peaks[1:]

def find_adpgPeaks(signals,fs,height):
    peaks = [0]
    width = int(fs * 0.7)

    #滑动窗口寻找峰值或者谷值
    for i in range(len(signals) - width * 2):
        try:
            subarr = signals[i:width + i] #获取当前窗口的信号序列

            middle = int(i + width / 2)  #或许当前窗口信号序列中间的索引值

            if (signals[middle] == max(subarr) and min(subarr) != max(subarr)):

                # 如果谷值点之后的采样点幅值也等于谷值，则排除这个点
                if (middle - peaks[-1] < width / 2):
                    continue

                if (signals[middle] < 0.5 * height):
                    continue

                peaks.append(middle)

        except ValueError:
            break


    return peaks[1:]

def findPV(signals,fs):
    valleys = findValleys(signals,fs)
    peaks = []
    vp = []

    for i in range(len(valleys) - 1):
        segs = signals[valleys[i] : valleys[i + 1]]
        a = np.argmax(segs)
        cur_peak = np.argmax(segs) + valleys[i]
        peaks.append(cur_peak)
        vp.append(valleys[i])
        vp.append(cur_peak)
    vp.append(valleys[-1])
    return vp,peaks,valleys

def GetSbpDbp(ABP,fs):
    vp,peaks,valleys = findPV(ABP,fs)
    SBP = 0
    DBP = 0

    if (len(peaks) > 1 and len(valleys) > 1):

        for i in range(len(peaks)):
            SBP += ABP[peaks[i]]

        for i in range(len(valleys)):
            DBP += ABP[valleys[i]]
    SBP = SBP / len(peaks)
    DBP = DBP / len(valleys)

    SBP = float(format(SBP, '0.1f'))
    DBP = float(format(DBP, '0.1f'))

    return SBP,DBP

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # person = "211_101.csv"
    # path = "G:\\mimic-database-1.0.0\\MIMIC_CSV\\55\\55_3.csv"
    # data = pd.read_csv(path, error_bad_lines=False, low_memory=False)
    # PPG = data.loc[:, ['ABP']]  #
    # PPG = np.ravel(PPG.values)

    path = "E:\\QQ文档\\weixin\\WeChat Files\\wxid_be3rh4xeu6zg22\\FileStorage\\File\\2021-06\\03_0.mp4.csv"
    data = pd.read_csv(path, error_bad_lines=False, low_memory=False)
    PPG = data.loc[:, ['nose_Green']]  #
    PPG = np.ravel(PPG.values)
    plt.plot(PPG)
    from Filter import bandpass_butter
    PPG = bandpass_butter(PPG,0.7,2.5,30)

    pv,peaks,valleys = findPV(PPG,30)
    plt.figure()
    plt.plot(PPG)

    for i in range(int(len(pv))):
        if (i % 2 == 0):
            plt.plot(pv[i],PPG[pv[i]],"r-o")
        else:
            plt.plot(pv[i], PPG[pv[i]], "g-o")

    plt.show()

