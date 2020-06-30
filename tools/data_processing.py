"""
preprocessing the pkl file data
and write it into csv file
"""

import os
os.sys.path.append('..')

from train import utils_wf
from sklearn.model_selection import train_test_split
import pandas as pd


def pkl2burst_csv( slice_threshold):
    "load pkl file, transform data to burst and write it into csv"

    x_path = '../data/NoDef/X_test_NoDef.pkl'
    out_path = '../data/NoDef/test_NoDef.csv'
    y_path = '../data/NoDef/y_test_NoDef.pkl'

    # x_path = '../data/NoDef/X_valid_NoDef.pkl'
    # out_path = '../data/NoDef/valid_NoDef.csv'
    # y_path = '../data/NoDef/y_valid_NoDef.pkl'

    # x_path = '../data/NoDef/X_train_NoDef.pkl'
    # out_path = '../data/NoDef/train_NoDef.csv'
    # y_path = '../data/NoDef/y_train_NoDef.pkl'

    x = utils_wf.load_pkl_data(x_path)
    y = utils_wf.load_pkl_data(y_path)
    X_new, y_new = utils_wf.data_preprocess(x, y)
    x_burst,x_burst_nopadding = utils_wf.burst_transform(X_new,slice_threshold)

    # utils_wf.size_distribution(x_burst_nopadding)

    burst_data = utils_wf.convert2dataframe(x_burst,y_new)
    utils_wf.write2csv(burst_data,out_path)



def merge_data():
    "merge train/test/valid data into one csv file processed source data (remove less than 50 packets and starting with incoming packet)"

    X_path = ['../data/NoDef/X_train_NoDef.pkl','../data/NoDef/X_test_NoDef.pkl','../data/NoDef/X_valid_NoDef.pkl']
    Y_path = ['../data/NoDef/y_train_NoDef.pkl','../data/NoDef/y_test_NoDef.pkl','../data/NoDef/y_valid_NoDef.pkl']
    out_path = '../data/NoDef/data_NoDef_processed.csv'

    X,Y = [],[]
    for x_path,y_path in zip(X_path,Y_path):
        X += utils_wf.load_pkl_data(x_path)
        Y += utils_wf.load_pkl_data(y_path)
    print('data instances after merged: {}'.format(len(Y)))

    "remove less than 50 packets and starting with incoming packet"
    X_new,Y_new = utils_wf.data_preprocess(X, Y)
    print('data instances after processed: {}'.format(len(Y_new)))

    data_new = utils_wf.convert2dataframe(X_new,Y_new)
    utils_wf.write2csv(data_new,out_path)


def tranform2burst(path,out_path,slice_threshold):
    "load processed source data from csv, transform to burst with certain fixed size and write it in csv"


    x,y = utils_wf.load_csv_data(path)
    x_burst, x_burst_nopadding = utils_wf.burst_transform(x, slice_threshold)
    burst_data = utils_wf.convert2dataframe(x_burst,y,mode='padding')
    utils_wf.write2csv(burst_data,out_path)



def extract_balance_data(path,num,outpath):
    "extract num instances for each class, write into csv"

    X,Y = utils_wf.extract_data_each_class(path,num)
    data = utils_wf.convert2dataframe(X,Y,mode='NoPadding')
    utils_wf.write2csv(data,outpath)




def test_train_split(path,out_train_path,out_test_path,test_size=0.2):
    "split processed data into train and test, write in csv"

    X,y = utils_wf.load_csv_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)

    X_test['label'] = y_test
    X_train['label'] = y_train
    X_test.to_csv(out_test_path, index=0)
    X_train.to_csv(out_train_path, index=0)

    print('No of instances of train data: {}'.format(len(X_train)))
    print('No of instances of test data: {}'.format(len(X_test)))


def normalize_data(path,out_path):
    "normlize data with MinMaxScaler to [-1,1], write it to csv"

    X,y = utils_wf.load_csv_data(path)

    X = utils_wf.data_normalize(X)
    X = pd.DataFrame(X)
    X['label'] = y
    X.to_csv(out_path,index=0)



def statistic_info(path):
    "the minimum number of instances for each class"

    _,Y = utils_wf.load_csv_data(path)
    statis = {}

    for y in Y:
        if y not in statis:
            statis[y] = 1
        else:
            statis[y] += 1

    values = statis.values()
    mini = min(values)
    print('the smallest number instances for each class among all class is: {}'.format(mini))



if __name__ == '__main__':

    #*************************************
    "source pkl to burst csv"
    # slice_threshold = 1024
    # pkl2burst_csv(slice_threshold)

    #*************************************
    "merge train/test/valid NoDef data as processed source data"
    # merge_data()


    #*************************************
    "statistic information"
    # path = '../data/NoDef/data_NoDef_processed.csv'
    # statistic_info(path)



    #*************************************
    "extract balance data from source data, num instances per class"
    # path = '../data/NoDef/data_NoDef_processed.csv'
    # outpath = '../data/NoDef/data_NoDef_balanced.csv'
    # num = 460       # num of instance per class
    # extract_balance_data(path,num,outpath)



    #*************************************
    "balanced processed source csv to burst csv"
    # path = '../data/NoDef/data_NoDef_balanced.csv'
    # out_path = '../data/NoDef/data_NoDef_burst.csv'
    # slice_threshold = 1024
    # tranform2burst(path,out_path,slice_threshold)

    # *************************************
    "Nomalize the burst data to [-1,1] and write it in csv"
    path = '../data/NoDef/data_NoDef_burst.csv'
    out_path = '../data/NoDef/data_norm_NoDef_burst.csv'
    normalize_data(path,out_path)

    # *************************************
    "split data into train/test data"
    path = '../data/NoDef/data_norm_NoDef_burst.csv'
    out_train_path = '../data/wf/train_NoDef_burst.csv'
    out_test_path = '../data/wf/test_NoDef_burst.csv'
    test_train_split(path,out_train_path,out_test_path,test_size=0.2)
