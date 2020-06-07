import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data


def load_data(path):
    "col_length: column length of each input"

    raw_data = pd.read_csv(path)
    x = raw_data.iloc[:,:-1]
    label = raw_data['label']

    return x, label


def data_preprocess(x, y,batch_size):
    """
    input: X, y ( X: dataframe, y: ndarray)
    output: tensor_loader. train_loader, test_loader
    """
    "normalize X"
    x = x / 1500.0

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
        shuffle = True,
    )

    return data_loader


def main(path,batch_size):
    x,y = load_data(path)
    data_loader = data_preprocess(x,y,batch_size)
    return data_loader


if __name__ == '__main__':

    batch_size = 1
    path = 'traffic_test.csv'
    x,label = load_data(path)
    data_loader = data_preprocess(x,label,batch_size)
    print(len(data_loader))
