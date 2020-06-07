
import pandas as pd
import pickle

object = pd.read_pickle(r'../data/NoDef/X_test_NoDef.pkl')
Y = pd.read_pickle(r'../data/NoDef/y_test_NoDef.pkl')

for x,y in zip(object,Y):
    print(x,y)
