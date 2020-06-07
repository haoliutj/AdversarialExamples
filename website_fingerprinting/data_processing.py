"""
preprocessing the pkl file data
and write it into csv file
"""

import utils_wf
from sklearn.model_selection import train_test_split


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
    "merge train/test/valid data into one csv file processed source data"

    X_path = ['../data/NoDef/X_train_NoDef.pkl','../data/NoDef/X_test_NoDef.pkl','../data/NoDef/X_valid_NoDef.pkl']
    Y_path = ['../data/NoDef/y_train_NoDef.pkl','../data/NoDef/y_test_NoDef.pkl','../data/NoDef/y_valid_NoDef.pkl']
    out_path = '../data/NoDef/data_NoDef_processed.csv'

    X,Y = [],[]
    for x_path,y_path in zip(X_path,Y_path):
        X += utils_wf.load_pkl_data(x_path)
        Y += utils_wf.load_pkl_data(y_path)
    print('data instances after merged: {}'.format(len(Y)))

    X_new,Y_new = utils_wf.data_preprocess(X, Y)
    print('data instances after processed: {}'.format(len(Y_new)))
    data_new = utils_wf.convert2dataframe(X_new,Y_new)
    utils_wf.write2csv(data_new,out_path)


def tranform2burst(path,out_path,slice_threshold):
    "load processed source data from csv, transform to burst and write it in csv"

    # path = '../data/NoDef/data_NoDef_processed.csv'
    # out_path = '../data/NoDef/data_NoDef_burst.csv'
    x,y = utils_wf.load_csv_data(path)
    x_burst, x_burst_nopadding = utils_wf.burst_transform(x, slice_threshold)
    burst_data = utils_wf.convert2dataframe(x_burst,y,mode='NoPadding')
    utils_wf.write2csv(burst_data,out_path)



def data_split(path,num):
    "extract num instances for each class, write into csv"

    X,Y = utils_wf.load_csv_data(path)
    X_new,Y_new = [],[]

    dic = {}
    for y in Y:
        if y not in dic:
            dic[y] = 1
        else:
            dic[y] += 1


    # for i,y in enumerate(Y):
    #     count = 1
    #     y_temp = y
    #     if Y
    #     while count <= num:




def train_test_split(path):
    "split processed data into train and test, write in csv"

    X,y = utils_wf.load_csv_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, stratify=y)

    X_test['label'] = y_test
    X_train['label'] = y_train
    X_test.to_csv('../data/NoDef/test_NoDef_burst.csv', index=0)
    X_train.to_csv('../data/DoDef/train_NoDef_burst.csv', index=0)

    print('No of instances of train data: {}'.format(len(X_train)))
    print('No of instances of test data: {}'.format(len(X_test)))



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

    "source pkl to burst csv"
    # slice_threshold = 512
    # pkl2burst_csv(slice_threshold)


    "merge train/test/valid NoDef data as processed source data"
    merge_data()


    "statistic information"
    # path = '../data/NoDef/data_NoDef_processed.csv'
    # statistic_info(path)

    "processed source csv to burst csv"
    path = '../data/NoDef/data_NoDef_processed.csv'
    out_path = '../data/NoDef/data_NoDef_burst.csv'
    slice_threshold = 512
    tranform2burst(path,out_path,slice_threshold)


