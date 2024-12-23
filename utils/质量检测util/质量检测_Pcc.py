import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matlab
import matlab.engine
from scipy.stats import pearsonr


def mapmaxmin(data,ymax, ymin):
    xmax = np.max(data)
    xmin = np.min(data)
    new_data = (ymax-ymin)*(data-xmin) / (xmax - xmin) + ymin
    return new_data


class Detect_seg():
    def __init__(self,data,eng,valleys,fs,segNum):
        self.data = data
        self.eng = eng
        self.fs = fs
        self.valleys = valleys
        self.segNum = segNum



    def peas_Seg(self):
        data = self.data
        valleys = self.valleys

        eng = self.eng


        peas_val = []

        forward_ptr = 1  # 初始化滑动指针，定为1是因为要结合之前的一段信号
        print("QA detect start....")
        while (forward_ptr < len(valleys) - 1):

            cur_seg = data[valleys[forward_ptr] : valleys[forward_ptr + 1]]
            peak_cur = np.max(cur_seg)
            valley_cur = np.min(cur_seg)

            before_seg = data[valleys[forward_ptr - 1]: valleys[forward_ptr]]
            peak_before = np.max(before_seg)
            valley_before = np.min(before_seg)

            val  = 0 # 初始化相邻两段信号的相关值

            if (np.abs(peak_cur - peak_before) > max(peak_cur - valley_cur, peak_before - valley_before) * 0.5):
                # 如果相邻两段信号峰值差距较大则置val为0
                val = 0

            elif (np.abs(len(cur_seg) - len(before_seg)) > len(before_seg) * 0.15):
                # 与上段信号长度相比，若差距较大则置0
                val = 0

            else:
                # 对前一个波形重采样
                before_seg = matlab.double(before_seg.tolist())
                before_seg = eng.resampleSig(before_seg,512)
                before_seg = np.array(before_seg)
                before_seg = np.ravel(before_seg)

                # 对当前波形重采样
                cur_seg = matlab.double(cur_seg.tolist())
                cur_seg = eng.resampleSig(cur_seg,512)
                cur_seg = np.array(cur_seg)
                cur_seg = np.ravel(cur_seg)

                val, _ = pearsonr(before_seg, cur_seg)

            forward_ptr += 1

            peas_val.append(val)


        return peas_val


    def goodQuality(self):
        peas_val = self.peas_Seg()
        segNum = self.segNum


        start_index = -1
        end_index = -1

        for i in range(len(peas_val)):
            if (peas_val[i] >= 0.8):
                if (start_index == -1 and end_index == -1):
                    start_index = i + 1
                    end_index = i + 2
                else:
                    pass

                end_index += 1

                if (end_index - start_index == segNum):
                    break
            else:
                start_index = -1
                end_index = -1

        if (end_index - start_index < segNum):
            start_index = -1
            end_index = -1

        return start_index,end_index








