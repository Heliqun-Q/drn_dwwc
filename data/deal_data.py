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
import matplotlib.pyplot as plt 
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
y_sample = [] 
x_sample = []
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

# x_train1= x_train;y_train_1=y_train

'''
def iteror_raw_data(data_path,data_mark):
    """ 
       读取.mat文件，返回数据的生成器：标签，样本数据。
  
       :param data_path：.mat文件所在路径
       :param data_mark："FE" 或 "DE"                                                   
       :return iteror：（标签，样本数据）
    """  
    # 标签数字编码
    labels = {"normal":0, "IR007":1, "IR014":2, "IR021":3, "OR007":4,
         "OR014":5, "OR021":6, "B007":7, "B014":8, "B021":9}
    # 列出所有文件
    filenams = os.listdir(data_path)
    # 逐个对mat文件进行打标签和数据提取
    for single_mat in filenams: 
        single_mat_path = os.path.join(data_path, single_mat)
        # 打标签
        for key, _ in labels.items():
            if key in single_mat:
                label = labels[key]
        # 数据提取
        file = loadmat(single_mat_path)
        for key, _ in file.items():
            if data_mark in key:
                #data = file[key]
                data = file[key].ravel()  # 2020/06/22， 降为一维
                        
        yield label, data
'''