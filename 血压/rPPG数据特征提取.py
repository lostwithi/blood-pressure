import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matlab
import matlab.engine
from scipy.signal import butter, filtfilt
from utils.WindowForPeaksOrValleys import findPeaks,find_adpgPeaks
from utils.FeatureUtil.特征提取 import PWA
from utils.质量检测util.质量检测_Pcc import Detect_seg
from utils.一阶微分和二阶微分 import getAll

def mapmaxmin(data, ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax - ymin) * (data - xmin) / (xmax - xmin) + ymin
    return new_data

def band_filter(data, low, high, fps):
    b, a = butter(3, [2*low/fps, 2*high/fps], "bandpass")
    signal_filter = filtfilt(b, a, data)
    return signal_filter

def valleys_index(data,peaks):
    valleys = []
    for i in range(len(peaks)):
        cur_peak = peaks[i]
        solding = cur_peak
        while(solding > 1):
            if (data[solding] <= data[solding - 1]):
                valleys.append(solding)
                break
            solding -= 1
    return valleys


if __name__ == '__main__':
    # 加载matlab引擎
    eng = matlab.engine.start_matlab()
    print("start")
    # 加载到指定文件进行matlab文件的使用
    eng.cd('D:\\zhang\\matlabFile')
    fs = 100

    rootArr = ["D:\zhang\数据集\BP_Data", "D:\zhang\数据集\帧"]
    for root in rootArr:
        persons = os.listdir(root)
        for person in persons:
            # personsPath例子 D:\zhang\数据集\BP_Data\cy
            personsPath = os.path.join(root,person)

            subroot = os.listdir(personsPath)
            if ( "rPPG" not in subroot):
                continue
            # personsPath例子 D:\zhang\数据集\BP_Data\cy\rPPG
            filesPath = os.path.join(personsPath,"rPPG")
            files = os.listdir(filesPath)
            for file in files:

                # fileRoot D:\zhang\数据集\BP_Data\cy\rPPG\cy_1.MP4.csv
                fileRoot = os.path.join(filesPath, file)
                print(fileRoot)
                data = pd.read_csv(fileRoot,engine='python',encoding="utf-8")
                BVP = data.loc[:, ['Cg_orgin']]
                BVP = BVP.values
                BVP = np.ravel(BVP)
                BVP = band_filter(BVP, 0.7, 3, fs)

                BVP_Peaks = findPeaks(BVP, fs)
                BVP_Valleys = valleys_index(BVP, BVP_Peaks)

                # plt.plot(BVP)
                # plt.plot(BVP_Peaks, BVP[BVP_Peaks] ,marker='o')
                # plt.show()
                if (len(BVP_Valleys) < 10):
                    continue
                QA = Detect_seg(BVP, eng, BVP_Valleys, fs, 5)
                start_index, end_index = QA.goodQuality()
                if (start_index == end_index):
                    print("No quality seg...")
                    continue
                # PPG特征
                goodVallays = BVP_Valleys[start_index:end_index]

                feature = []
                # feature.extend([file])
                PWAFeature = PWA(BVP,goodVallays,Flag=True)
                # feature.extend(PWAFeature)
                # info = pd.DataFrame([feature])
                # info.to_csv(
                #    'MultiFeature.csv',
                #     mode='a', header=False, index=False)
                print()
