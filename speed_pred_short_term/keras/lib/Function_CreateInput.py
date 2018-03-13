import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *

def shapeback(Y):
    YY = np.reshape(Y[len(Y)%288:,:],(len(Y)//288,288,Y.shape[1]))
    return np.swapaxes(YY,1,2)

def create_dataset(Speed, Speed_y, look_back=15, mode='uni'):
    
    dataX,dataY = [],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,0:1]
            dataset_y = Speed_y[j,:,0:1]
            
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset_y[i + look_back, :])
            
            dataX.append(np.array(dataX_))
            dataY.append(np.array(dataY_))

    if mode == 'multi':
    
        for j in range(len(Speed)):
            dataX_,dataY_ = [],[]
            dataset = Speed[j,:,:]
            dataset_y = Speed_y[j,:,:]
            
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                dataX_.append(a)
                dataY_.append(dataset_y[i + look_back, :])
            
            dataX.append(np.array(dataX_))
            dataY.append(np.array(dataY_))


    return np.array(dataX), np.array(dataY)

def create_dataset_historyAsFeature(Speed, Speed_y, look_back=15, mode='uni'):
    
    dataX,dataY = [],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)-look_back):
            dataX_,dataY_ = [],[]
            dataset = Speed[j + look_back,:,0:1]
            dataset_y = Speed_y[j + look_back,:,0:1]
            prevdata = Speed[j:(j + look_back),:,0:1]
            
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i + look_back,:]
                dataX_.append(np.hstack([a,b]))
                dataY_.append(dataset_y[i + look_back, :])
            
            dataX.append(np.array(dataX_))
            dataY.append(np.array(dataY_))

    return np.array(dataX), np.array(dataY)

def create_dataset_historyAsSecondInput(Speed, Speed_y, look_back=15,look_back_days=6, mode='uni'):
    
    dataX1,dataX2,dataY = [],[],[]
    
    if mode == 'uni':
        
        for j in range(len(Speed)-look_back_days):
            dataX1_,dataX2_,dataY_ = [],[],[]
            dataset = Speed[j + look_back_days,:,0:1]
            dataset_y = Speed_y[j + look_back_days,:,0:1]
            prevdata = Speed[j:(j + look_back_days),:,0:1]
            
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i+look_back,:]
                dataX1_.append(a)
                dataX2_.append(b)
                dataY_.append(dataset_y[i + look_back, :])
            
            dataX1.append(np.array(dataX1_))
            dataX2.append(np.array(dataX2_))
            dataY.append(np.array(dataY_))

    if mode == 'multi':
    
        for j in range(len(Speed)-look_back_days):
            dataX1_,dataX2_,dataY_ = [],[],[]
            dataset = Speed[j + look_back_days,:,:]
            dataset_y = Speed_y[j + look_back_days,:,:]
            prevdata = Speed[j:(j + look_back_days),:,:]
            
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back), :]
                b = prevdata[:,i+look_back,:]
                dataX1_.append(a)
                dataX2_.append(b)
                dataY_.append(dataset_y[i + look_back, :])
            
            dataX1.append(np.array(dataX1_))
            dataX2.append(np.array(dataX2_))
            dataY.append(np.array(dataY_))

    return np.array(dataX1), np.array(dataX2), np.array(dataY)

if __name__ == '__main__':

    train_speed, _, _, _, _ = get_certain_dayofweek(Speed[:334],dayofweek = 0)
    test_speed = train_speed[-1:]
    train_speed = train_speed[:-1]

    print('train_speed.shape = ',train_speed.shape)
    print('test_speed.shape = ',test_speed.shape)
    look_back = 15
    mode = 'uni'
    train_speed_x,train_speed_y = create_dataset(train_speed,train_speed, look_back, mode)
    test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)
    print('look_back = ',look_back)
    print('mode = ',mode)
    print('train_speed_x.shape = ',train_speed_x.shape)
    print('train_speed_y.shape = ',train_speed_y.shape)
    print('test_speed_x.shape = ',test_speed_x.shape)
    print('test_speed_y.shape = ',test_speed_y.shape)
