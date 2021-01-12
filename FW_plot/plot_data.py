# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 24:17
# @Author  : Xin Deng
# @FileName: plot_data.py


import seaborn as sns
import numpy as np
import sys, time, random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from numpy import linalg as LA
import scipy.sparse
from scipy.stats import rv_continuous


n = 128
m = np.int32(np.linspace(32, 4 * (n - 32), 15))
s = np.int32(np.linspace(8, n - 8, 15))
average_num = 5.
data = np.load('./results/list_res_100.npy')
# data = np.delete(data, 24, 1)
data = np.sum(data, 2) / average_num
print(data)
res = np.zeros((15, 15))

ratio1 = s / n
ratio2 = m / n

for i in range(15):
    res[i, :] = data[:, 14 - i].T
    res[i, :] = res[i, ::-1]

print(res)

plt.imshow(res,
           extent=(np.amin(ratio1), np.amax(ratio1), np.amin(ratio2), np.amax(ratio2)),
            cmap=cm.hot,
           aspect='auto')
plt.colorbar()
plt.savefig('./results/n_100')
plt.show()
