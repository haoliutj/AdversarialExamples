from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pandas as pd
import torch
import torch.utils.data as Data
import numpy as np
import copy


def incoming_data_extract(data, slice_threshold=256):
    "remove outgoing data for each traffic"
    "padding to the same length with 0"
    "output dataframe"
    "observe the length of most of the traffic data are smaller than 248, so slice at 248 "
    "256 choose 256/2/2=64, easy to calculate the size in CNN"
    #outgoing package size positice value
    data_group = []
    # temp1 = []
    for i in range(len(data)):
        row_i = data.iloc[i]
        temp = []
        for x in row_i:
            if x > 0:
                temp.append(x)
        # temp1.append(len(temp))
        data_group.append(temp[:slice_threshold])
    df = pd.DataFrame(data_group) # after convert list to DF, all short lists will append with "NaN' to the same length
    df = df.fillna(0)       # all those 'NaN' are padded to same length, replace 'NaN' with 0
    return df


def load_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)

    "encode labels"
    y_raw = raw_data.iloc[:,0]
    encoder = preprocessing.LabelEncoder()
    labels = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    labels = np_utils.to_categorical(labels,num_classes=num_classes)

    "processing data"
    X = raw_data.iloc[:,1:]
    X = incoming_data_extract(X)

    "train test data"
    X_train,X_test,y_train,y_test = train_test_split(X,labels,train_size=0.94,shuffle=True,stratify=labels) # stratify = y, split data according to proportion

    # print(y_test)

    return X_train,y_train,X_test,y_test,num_classes


def write2file(X_train,y_train,X_test,y_test):
    y_train = np.argmax(y_train)
    y_test = np.argmax(y_test)
    # print(y_test)
    X_train['label'] = y_train
    X_test['label'] = y_test
    X_test.to_csv('test_data.csv',index=False)


def data_preprocess(X_train,y_train,X_test,y_test,Config):
    """
    input: X_train,y_train,X_test,y_test ( X: dataframe, y: ndarray)
    output: tensor_loader. train_loader, test_loader
    """
    "normalize X"
    X_train = X_train / 1500.0
    X_test = X_test / 1500.0

    " add dimension, df to ndarray "
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)


    "ndarray to tensor:"
    Y_train_tensor = torch.Tensor(y_train)
    Y_test_tensor = torch.Tensor(y_test)
    X_train_tensor = torch.Tensor(X_train)
    X_test_tensor = torch.Tensor(X_test)

    # labels from vector to index
    Y_train_tensor = torch.max(Y_train_tensor,1)[1]
    Y_test_tensor = torch.max(Y_test_tensor,1)[1]


    "build input dataformat in pytorch"
    train_dataset = Data.TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = Data.TensorDataset(X_test_tensor, Y_test_tensor)


    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=Config.test_batch_size,
        shuffle=True,
    )


    return train_loader, test_loader


def main(path,Config):
    X_train, y_train, X_test, y_test, _ = load_data(path)
    train_loader,test_loader = data_preprocess(X_train,y_train,X_test,y_test,Config)
    return train_loader,test_loader



# if __name__ == '__main__':
#     path = '../generic_class.csv'
#
#     class Config:
#         batch_size = 64
#         test_batch_size = 1
#
#     train_loader,test_loader = main(path,Config)