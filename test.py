import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, Normalizer

a = [1,2,3,4,5,-1,-2,-3,-4,-5,7]
r = [0.1,0.2,-0.1,0.001,0.03,-0.004,-0.1,0.3,0.1,-0.3,-0.1]
a = np.array(a)
# b = a.reshape((len(a),1))
# print(b.shape)

b= [[[1,-5,14],[0,-8,16],[-21,-3,19],[34,12,-21]]]
b = np.array(b)
# b = b.reshape(1,-1)
print('original data',b)
print(b.shape)

# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = scaler.fit(b)

# scaler = Normalizer(norm='l2').fit(b)
# norm = scaler.transform(b)
# print(norm)



class normalizer:
    """
    input data: ndarray, 1-d in row
    """
    def __init__(self,data,mode='l2'):

        if isinstance(data, torch.Tensor):
            data = data.data.cpu().numpy()
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        self.input_shape = data.shape
        self.data = np.transpose(data.squeeze())

        if mode == 'l1':
            self.w = sum(abs(self.data))
        elif mode == 'l2':
            self.w = np.sqrt(sum(self.data ** 2))


    def Normalizer(self):
        x_norm = self.data / self.w
        x_norm = np.transpose(x_norm).reshape(self.input_shape)

        return x_norm


    def inverse_Normalizer(self,x_norm):
        x_norm = np.transpose(x_norm.squeeze())
        x = x_norm * self.w
        x = np.transpose(x).reshape(self.input_shape)

        return x


normalization = normalizer(b,'l2')
x_norm = normalization.Normalizer()
x = normalization.inverse_Normalizer(x_norm )
print('norm:',x_norm)
print('inverse',x)
print('round',np.around(x))



# normalized = scaler.transform(b)
# print('normalized data',normalized)
#
#
# normalized = normalized + np.array([0.8])
# print('add values',normalized)
# d = scaler.inverse_transform(normalized)
# print('inverse after adding values',d)

