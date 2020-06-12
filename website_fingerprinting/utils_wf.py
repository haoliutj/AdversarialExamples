import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler



def params(num_class,input_size):
    "hyperparameters for target model"

    print('input_size: {}'.format(input_size))

    return {
        'conv1_input_channel': 1,
        'conv2_input_channel': 128,
        'conv3_input_channel': 128,
        'conv4_input_channel': 64,
        'conv1_output_channel': 128,
        'conv2_output_channel': 128,
        'conv3_output_channel': 64,
        'conv4_output_channel': 256,
        'kernel_size1': 7,
        'kernel_size2': 19,
        'kernel_size3': 13,
        'kernel_size4': 23,
        'stride1': 1,
        'stride2': 1,
        'stride3': 1,
        'stride4': 1,
        'padding1': 3,
        'padding2': 9,
        'padding3': 6,
        'padding4': 11,
        'drop_rate1': 0.1,
        'drop_rate2': 0.3,
        'drop_rate3': 0.1,
        'drop_rate4': 0.0,
        'pool1': 2,
        'pool2': 2,
        'pool3': 2,
        'pool4': 2,
        'num_classes': num_class,
        'input_size':input_size
    }




def load_pkl_data(path):
    "load pickle file"
    data = pd.read_pickle(path)

    return data



def burst_transform(data,slice_threshold):
    "transform binary format to burst format"

    "dataframe to list"
    if type(data) == list:
        pass
    else:
        data = data.values.tolist()

    burst_data = []
    burst_nopadding = []

    for x in data:
        burst_i = []
        count = 1
        temp = x[0]

        for i in range(1,len(x)):
            if temp == 0:
                print('traffic start with 0 error')
                break
            elif x[i] == 0:
                burst_i.append(count*temp)
                break
            elif temp == x[i]:
              count += 1
            else:
                burst_i.append(count*temp)
                temp = x[i]
                count = 1
        burst_data.append(burst_i[:slice_threshold])
        burst_nopadding.append(burst_i)
    return burst_data,burst_nopadding



def size_distribution(data):
    "plot size distribution in different ways"
    "get lenght of each data"
    length = []
    for x in data:
        length.append(len(x))

    "plot cdf"
    hist,bin_edges = np.histogram(length)
    cdf = np.cumsum(hist/sum(hist))
    plt.figure(0)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel('number of bursts')
    plt.ylabel('Fraction')
    plt.savefig('./fig/size_cdf.pdf')
    plt.show()

    "plot cdf and pdf in one fig"
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    plt.figure(1)
    plt.bar(bin_edges[1:], hist / max(hist), width=width, color='#5B9BD5')
    plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
    plt.xlabel('number of bursts')
    plt.ylabel('Fraction')
    plt.savefig('./fig/size_cdf_pdf.pdf')
    plt.show()


    "get statistic info about the length in dictionary format"
    size_dic = {}
    for x in length:
        if x not in size_dic:
            size_dic[x] = 1
        else:
            size_dic[x] +=1

    "obtain fraction of each value in dic"
    keys = size_dic.keys()
    values = np.array(list(size_dic.values()))
    values_fraction = np.array(list(values))/sum(values)

    "present in bar figure"
    plt.figure(2)
    plt.bar(keys,values)
    plt.xlabel('number of bursts')
    plt.ylabel('numer of traces')
    plt.savefig('./fig/size_distribution.pdf')
    plt.show()

    return size_dic



def data_preprocess(X,y):
    """
    removing any instances with less than 50 packets
    and the instances start with incoming packet
    """
    X_new = []
    y_new = []
    length = []
    for x in X:
        x = [i for i in x if i != 0]
        length.append(len(x))
    for i,x in enumerate(length):
        if x >= 50 and X[i][0] == 1:
            X_new.append(X[i])
            y_new.append(y[i])

    print('No of instances after processing: {}'.format(len(y_new)))

    return X_new,y_new



def convert2dataframe(x,y,mode='padding'):
    "convert list to dataframe format, merge x and y in one df"
    df = pd.DataFrame(x)

    if mode == 'padding':
        df = df.fillna(0)
    else:
        pass
    df['label'] = y

    return df



def write2csv(data,outpath):
    " dataframe write to csv"

    data.to_csv(outpath,index=0)



def load_csv_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)
    x = raw_data.iloc[:,:-1]
    label = raw_data['label']

    return x, label



def data_normalize(data):
    """
    normalize data with MinMaxScaler [-1,1]
    :param data: ndarray
    :return: ndarray
    """

    if not isinstance(data,np.ndarray):
        data = data.to_numpy()

    input_shape = data.shape
    data = data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    data = data.reshape(input_shape)

    return data


def inverse_normalize(input_data):
    """

    :param input_data: ndarray
    :return: ndarray
    """

    if type(input_data) == torch.Tensor:
        input_data = input_data.data.cpu().numpy()

    input_shape = input_data.shape
    input_data = input_data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    input_data = scaler.inverse_transform(input_data)
    input_data = input_data.reshape(input_shape)

    return input_data



def Data_loader(x, y,batch_size):
    """
    input: X, y ( X: dataframe, y: ndarray)
    output: tensor_loader. train_loader, test_loader
    """


    " add dimension, df to ndarray "
    x = np.expand_dims(x, axis=1)

    "ndarray to tensor:"
    x_tensor = torch.Tensor(x)
    y_tensor = torch.Tensor(y).long()

    "build input dataformat in pytorch"
    dataset = Data.TensorDataset(x_tensor,y_tensor)

    data_loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = False,
    )

    return data_loader



def load_data_main(path,batch_size):
    x,y = load_csv_data(path)
    data_loader = Data_loader(x,y,batch_size)

    return data_loader



def extract_data_each_class(path,num):
    "extract num instances for each class,return data in list"

    X,Y = load_csv_data(path)
    X_new, Y_new = [],[]

    Y_set = set(Y)


    for y_set_i in Y_set:
        temp = y_set_i
        count = 1
        for i,y in enumerate(Y):
            if y == temp and count <= num:
                X_new.append(X.iloc[i,:])
                Y_new.append(y)
                count += 1
                if count > num:
                    break
    print('{} instances per class, {} classes in toal.'.format(num,len(Y_set)))
    return X_new,Y_new




