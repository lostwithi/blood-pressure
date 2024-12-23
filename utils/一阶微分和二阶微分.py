import numpy as np


def dPPG(PPG):
    res = []
    for i in range(len(PPG)):
        if (i == 0):
            continue
        res.append( (PPG[i] - PPG[i - 1] ) )
    return res

def APG(dPPG):
    res = []
    for i in range(len(dPPG)):
        if (i == 0):
            continue
        res.append( (dPPG[i] - dPPG[i - 1]) )
    return res

def getAll(PPG):
    dPG = dPPG(PPG)
    dPG = np.array(dPG)
    apg = APG(dPG)
    apg = np.array(apg)
    return dPG[1:],apg