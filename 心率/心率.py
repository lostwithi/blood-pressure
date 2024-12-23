import h5py
import numpy as np
from DecomMethod import DecomMethod
import matlab
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine
from scipy.signal import find_peaks, butter, filtfilt
from HrCompute import get_hr
import os
import time

eng = matlab.engine.start_matlab()
print("start")
# 加载到指定文件进行matlab文件的使用
eng.cd('D:\zhang\matlabFile')


def quality_judge_filter(data, fps):
    import utils.质量检测util.质量筛选 as QM
    Flag, level, start_index, end_index = QM.sliding_window_for_optimal_segment_index(data=data, window_length=15,                                                                               window_step=1, fps=fps)
    if Flag == False:
        return []
    data = data[start_index: end_index]
    return data


def read_personsFile(path):
    data = pd.read_csv(path, engine='python', encoding="utf-8")
    files = data.loc[:, : ]
    # BVP = data.loc[:, :]
    files = files.values
    files = np.ravel(files)
    return files


def band_filter(data, low, high, fps):
    b, a = butter(3, [2*low/fps, 2*high/fps], "bandpass")
    signal_filter = filtfilt(b, a, data)
    return signal_filter

def read_rPPG(path):
    data = pd.read_csv(path, engine='python', encoding="utf-8")
    BVP = data.loc[:, ["rPPG_orgin"]]
    # BVP = data.loc[:, :]
    BVP = BVP.values
    BVP = np.ravel(BVP)
    return BVP

def rPPGSignalProcess(path, fs, eng, methodType, startSec = None, endSec = None):
    data = pd.read_csv(path, engine='python', encoding="utf-8")
    BVP = data.loc[:, ["rPPG_orgin"]]
    # BVP = data.loc[:, :]
    BVP = BVP.values
    BVP = np.ravel(BVP)
    if ( startSec != None and endSec != None ):
        BVP = BVP[fs * startSec: fs * endSec]
    # BVP = quality_judge_filter(BVP, fs)
    if (len(BVP) == 0): return []
    method = DecomMethod(fs, methodType, eng)
    signal = method.getDecomSignal(BVP.copy())

    # plt.plot(signal)
    # plt.show()
    return signal

def selfDataHR(methodType):
    root = "D:\chen\论文\IAM\data\DealedWithEVM\Freq-band-0.5-to-2"

    fs = 30

    dataPath = "E:\\30人\\rPPGhr"

    method = DecomMethod(fs, methodType, eng)

    scens = os.listdir(root)
    scens = ["正"]
    for scen in scens:

        scenPath = os.path.join(root, scen)
        if os.path.isfile(scenPath):
            continue

        persons = os.listdir(scenPath)

        for person in persons:



            path = os.path.join(scenPath, person)

            if os.path.isdir(path):
                continue
            print(path)
            data = pd.read_csv(path, engine='python', encoding="utf-8")

            Red = data.loc[:, ["Red"]]
            # BVP = data.loc[:, :]
            Red = Red.values
            Red = np.ravel(Red)
            Red = DeleteNanNum(Red)


            Green = data.loc[:, ["Green"]]
            # BVP = data.loc[:, :]
            Green = Green.values
            Green = np.ravel(Green)
            Green = DeleteNanNum(Green)

            Blue = data.loc[:, ["Blue"]]
            # BVP = data.loc[:, :]
            Blue = Blue.values
            Blue = np.ravel(Blue)
            Blue = DeleteNanNum(Blue)

            personNum = person[0:2]
            pieceNum = int(person[5:6])

            firstPiece_R = Red[: 30 * fs]
            firstPiece_R = band_filter(firstPiece_R, 0.7, 3, fs)
            firstPiece_G = Green[ : 30 * fs ]
            firstPiece_G = band_filter(firstPiece_G, 0.7, 3, fs)
            firstPiece_B = Blue[: 30 * fs]
            firstPiece_B = band_filter(firstPiece_B, 0.7, 3, fs)

            firstPieceInput = []
            firstPieceInput.extend([firstPiece_R])
            firstPieceInput.extend([firstPiece_G])
            firstPieceInput.extend([firstPiece_B])


            t0 = time.time()

            firstPiece = method.getDecomSignal(firstPieceInput)


            # firstPiece = quality_judge_filter(firstPiece, fs)
            t1 = time.time()
            timeUsed.append(t1 - t0)

            if (len(firstPiece) > 0):
                first_fft_hr, first_time_hr = get_hr(firstPiece, fs)
                firstPieceName = person[:4] + "_" + str( (pieceNum*2 + 0) * 30) + "_" + str( (pieceNum*2 + 1) * 30)

                save_clip = []
                save_clip.append(firstPieceName)
                save_clip.append(first_time_hr)
                save_clip.append(first_fft_hr)


                # info = pd.DataFrame([save_clip])
                # info.to_csv(
                #     os.path.join(dataPath, methodType, methodType + "_rPPG" + "_" + scen + ".csv"),
                #     mode='a', header=False, index=False, encoding="utf_8_sig")




            secondPiece_R = Red[ 30 * fs :  ]
            secondPiece_R = band_filter(secondPiece_R, 0.7, 3, fs)
            secondPiece_G = Green[ 30 * fs :  ]
            secondPiece_G = band_filter(secondPiece_G, 0.7, 3, fs)
            secondPiece_B = Blue[ 30 * fs :  ]
            secondPiece_B = band_filter(secondPiece_B, 0.7, 3, fs)

            secondPieceInput = []
            secondPieceInput.extend([secondPiece_R])
            secondPieceInput.extend([secondPiece_G])
            secondPieceInput.extend([secondPiece_B])

            t0 = time.time()
            secondPiece = method.getDecomSignal(secondPieceInput)


            # secondPiece = quality_judge_filter(secondPiece, fs)

            t1 = time.time()
            timeUsed.append(t1 - t0)

            if (len(secondPiece) > 0):
                second_fft_hr, second_time_hr = get_hr(secondPiece, fs)
                secondPieceName = person[:4] + "_" + str( (pieceNum*2 + 1) * 30) + "_" + str( (pieceNum*2 + 2) * 30)

                save_clip = []
                save_clip.append(secondPieceName)
                save_clip.append(second_time_hr)
                save_clip.append(second_fft_hr)


                # info = pd.DataFrame([save_clip])
                # info.to_csv(
                #     os.path.join(dataPath, methodType, methodType + "_rPPG" + "_" + scen + ".csv"),
                #     mode='a', header=False, index=False, encoding="utf_8_sig")

    pass

def DeleteNanNum(sigs):
    signals = sigs
    for i in range(len(signals)):
        if (np.isnan(signals[i])):
            signals[i] = 0

    return signals

if __name__ == '__main__':

    timeUsed = []

    methodType = "acmd"
    selfDataHR(methodType)

    print(np.average(timeUsed) * 1000)
    print(np.std(timeUsed) * 1000 )


    pass





