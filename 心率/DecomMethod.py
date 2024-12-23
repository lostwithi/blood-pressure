from scipy.signal import butter
from scipy.signal import filtfilt
import matlab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matlab.engine
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy import signal
from sklearn.decomposition import FastICA
from PyEMD import EEMD,EMD
from utils.VMD import Vmd

def band_filter(data, low, high, fps):
    b, a = butter(3, [2*low/fps, 2*high/fps], "bandpass")
    signal_filter = filtfilt(b, a, data)
    return signal_filter

def mapmaxmin(data,ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax-ymin)*(data-xmin) / (xmax - xmin) + ymin
    return new_data


def FreAnalysis(signals,fs):
    '''

    :param signals: 输入信号
    :param fs: 采样频率
    :return: 返回频率和对应幅值
    '''
    iSampleCount = len(signals)
    xFFTAmplitude = np.abs(np.fft.rfft(signals) / iSampleCount) * 2
    xFreqs = np.linspace(0, fs / 2, int(iSampleCount / 2) + 1)
    return xFreqs,xFFTAmplitude


def recon_sig_max_pearson(IPPG,IMFs,fs,low,high):

    fres, Amplitude = FreAnalysis(IMFs[0], fs)
    max_peak = fres[np.argmax(Amplitude)]

    recon_index = 0
    pear = abs(np.corrcoef([IMFs[0], IPPG])[0][1])

    flag = False

    for i in range(len(IMFs)):
        fres, Amplitude = FreAnalysis(IMFs[i], fs)
        cur_high_fre  = fres[np.argmax(Amplitude)]
        if (i == len(IMFs) - 1):
            continue
        if (i == 0):
            max_peak = cur_high_fre
        elif(cur_high_fre > max_peak):
            max_peak = cur_high_fre
        # elif (cur_high_fre < max_peak and flag):
        #     break

        if (cur_high_fre > low and cur_high_fre < high):

            # IMFs[i] = mapmaxmin(IMFs[i], 1, -1)
            cur_pear,_ =  pearsonr(IMFs[i], IPPG)
            # print("第" + str(i + 1) + "分量与原信号相关系数：" + str(cur_pear) + "   最大峰值：" + str(cur_high_fre) )

            if (flag == False):
                max_peak = cur_high_fre
                pear = cur_pear
                recon_index = i
                flag = True

            if (cur_pear > pear):
                pear = cur_pear
                recon_index = i

    return IMFs[recon_index]




def acmd_def(data, fps, eng):
    iPPG = data
    iPPG = mapmaxmin(iPPG, 1, -1)

    # ACMD的参数
    alpha0 =  0.000025  #1e-4
    tol = 1e-8
    re = 0.01  # 当当前分解的信号的能量小于前一个能量的0.01，则停止分解;可通过调整该值大小来控制分解的数量
    Sig = iPPG
    Sig = matlab.double(Sig.tolist())  # 转换成MATLAB的数据格式

    eIMFs = eng.iter_ACMD1(Sig, fps, alpha0, tol, re)
    eIMFs = np.array(eIMFs)  # 将结果转换为numpy类型
    return eIMFs


def vmd_def(data):
    data = mapmaxmin(data, 1, -1)
    K = 5
    alpha = 2000
    tau = 0
    vmd = Vmd(K, alpha, tau, tol=1e-7, maxIters=500, eps=1e-9)
    eIMFs, omega_K = vmd(data)
    return eIMFs


def eemd_def(data):
    data = mapmaxmin(data, 1, -1)
    method = EEMD()
    eIMFs = method.eemd(np.asarray(data))
    return eIMFs

def emd_def(data):
    data = mapmaxmin(data, 1, -1)
    method = EMD()
    eIMFs = method.emd(np.asarray(data))
    return eIMFs



def ica_def(data_R, data_G, data_B):
    D = np.asarray([data_R, data_G, data_B]).T
    fast_ica = FastICA(n_components=3)  # 独立成分为2个
    eIMFs = fast_ica.fit_transform(D).T  # SrT为解混后的2个独立成分，shape=[m,n]
    max_index = 0
    max_pearson = abs(np.corrcoef([eIMFs[0], data_G])[0][1])
    # 根据相关系数确定输出的信号

    for single_imf_index in np.arange(1, len(eIMFs), 1):
        cur_pearson = abs(np.corrcoef([eIMFs[single_imf_index], data_G])[0][1])
        if cur_pearson > max_pearson:
            max_pearson = cur_pearson
            max_index = single_imf_index
    return eIMFs[max_index]

def quality_judge_filter(data, fps):
    import utils.质量检测util.质量筛选 as QM
    Flag, level, start_index, end_index = QM.sliding_window_for_optimal_segment_index(data=data, window_length=15,                                                                               window_step=1, fps=fps)
    if Flag == False:
        return []
    data = data[start_index: end_index]
    return data

def DeleteNanNum(sigs):
    signals = sigs
    for i in range(len(signals)):
        if (np.isnan(signals[i])):
            signals[i] = 0

    return signals

class DecomMethod():
    def __init__(self, fs, methodType, eng):
        self.methodType = methodType
        self.fs = fs
        self.eng = eng

    def getDecomSignal(self, data):
        r = data[0]
        g = data[1]
        b = data[2]
        r = mapmaxmin(r, 1, -1)
        r = band_filter(r, 0.3, 6, self.fs)
        r = DeleteNanNum(r)

        g = mapmaxmin(g, 1, -1)
        g = band_filter(g, 0.3, 6, self.fs)
        g = DeleteNanNum(g)

        b = mapmaxmin(b, 1, -1)
        b = band_filter(b, 0.3, 6, self.fs)
        b = DeleteNanNum(b)

        eIMFs = None
        if ( self.methodType == None or self.methodType == 'acmd' ):
            eIMFs = acmd_def(g, self.fs, self.eng)
            eIMFs = eIMFs[0:-1]
        elif (self.methodType == 'vmd'):
            eIMFs = vmd_def(g)
        elif (self.methodType == 'eemd'):
            eIMFs = eemd_def(g)
        elif (self.methodType == 'emd'):
            eIMFs = emd_def(g)
        elif (self.methodType == 'fastica'):
            if ( r.any() == 0 and g.any() == 0 and b.any() == 0 ):
                return g
            resultSignal = ica_def(r,g,b)
            return resultSignal
        resultSignal = recon_sig_max_pearson(g, eIMFs, self.fs, 0.7, 3)


        return resultSignal