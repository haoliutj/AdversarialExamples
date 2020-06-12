import pandas as pd
import torch
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
#
# # a = [1,2,3,4,5,-1,-2,-3,-4,-5,7]
# # r = [0.1,0.2,-0.1,0.001,0.03,-0.004,-0.1,0.3,0.1,-0.3,-0.1]
# # a = np.array(a)
# # b = a.reshape((len(a),1))
# # print(b.shape)
#
# b= [[[1,2,3],[3,4,5]]]
# b = np.array(b)
# b = b.reshape(-1,1)
# print(b)
# print(b.shape)
# print(b.reshape(1,2,3))
#
# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = scaler.fit(b)
# print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
#
# normalized = scaler.transform(b)
# for i in range(len(normalized)):
#     print(normalized[i])
#
# print(normalized.shape)
#
#
# print('\n')
#
#
# c = normalized.reshape(len(normalized))
# print(c.shape)
# print(c)
#
# f = np.array(r)
# print(f)
# t = c+f
# print(t)
# t = t.reshape(len(t),1)
# d = scaler.inverse_transform(t)
# print(d)
#
# k = a.reshape(1,2,1)
# print(k)
#
g = [1,2,3,4]
h = np.array(g)
print(isinstance(h,np.ndarray))