# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 21:22:50 2021

@author: 17845
"""


import os
from scipy.io  import loadmat
import numpy as np
import pywt
import random
data_path = '.\gear_data_48'

DE_data = []
FE_data = []
filenams = os.listdir(data_path)
# 逐个对mat文件进行打标签和数据提取
for single_mat in filenams: 
    print(single_mat)
    single_mat_path = os.path.join(data_path, single_mat)
    file = loadmat(single_mat_path)
    for key in file.keys():
        if key.find('DE') != -1:
            DE_data.append(file[key].reshape(len(file[key])))
        if key.find('FE') != -1:
            FE_data.append(file[key].reshape(len(file[key])))

del DE_data[3]; del FE_data[3]
# 求原始振动数据的小波包分解特征
y_sample = [] 
x_sample = []
# 窗移设置为256
length = 256

for i in range(len(DE_data)):
    len_sample = int((len(DE_data[i])-4096)/length)
    wavelet_coefs = np.zeros([len_sample, 1, 64, 64])
    for num_sample in range(len_sample):
        wp = pywt.WaveletPacket(DE_data[i][length*i: length*i + 4096], wavelet='db1', mode='symmetric').get_level(6,order='freq')
        y_sample.append(i)
        for _ in range(64):
            wavelet_coefs[num_sample, 0, _, :] = wp[_].data
    x_sample.append(wavelet_coefs)      

x_train = x_sample[0]
for i in range(1,len(x_sample)):
    x_train = np.concatenate((x_train, x_sample[i]))
y_train = np.array(y_sample)

#打乱数据集
np.random.seed(1)
index = list(range(0, len(y_train)))
np.random.shuffle(index)
x_data = x_train[index]
y_data = y_train[index]
np.save('x_origin.npy', x_train)
np.save('y_origin.npy', y_train)
np.save('x_data.npy', x_data)
np.save('y_data.npy', y_data)



#对训练集和验证集进行切分
mm = range(len(y_train))
index = random.sample(mm, 1500)

x_valid = x_train[index]
y_valid = y_train[index]
x_tra = np.delete(x_train, index, axis=0)
y_tra = np.delete(y_train, index, axis=0)
np.save('x_train.npy', x_tra)
np.save('y_train.npy', y_tra)
np.save('x_valid.npy', x_valid)
np.save('y_valid.npy', y_valid)

