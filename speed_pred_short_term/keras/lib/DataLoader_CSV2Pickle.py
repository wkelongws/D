import pickle
import numpy as np
import pandas as pd
import csv

def load_data(data,datafolder='/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/data/Data_2016_DES_I235E/'):
    X_train = np.zeros((len(data),15,288,10))
    y_train = np.zeros((len(data),15,1440,3))
    for i in range(len(data)):
        with open(datafolder+'CSVs/'+data['X'][i], 'r') as f:
            X = list(csv.reader(f, delimiter=","))
            X = np.asarray(X)
            channels = np.unique(X[:,0])
            for channel in channels:
                index=X[:,0] == channel
                X_train[i,:,:,int(channel)] = X[index,1:]
        with open(datafolder+'Traffic_CSVs/'+data['y'][i], 'r') as f:
            y = list(csv.reader(f, delimiter=","))
            y = np.asarray(y)
            channels = np.unique(y[:,0])
            for channel in channels:
                index=y[:,0] == channel
                y_train[i,:,:,int(channel)] = y[index,1:]
#             index=y[:,0] == '0'
#             y_train[i,:,:,0] = y[index,1:]
    return X_train,y_train

def convert_zero_2_mean(y_train):
    y_train[y_train==0] = np.nan
    avg = np.nanmean(y_train, axis=0)
    arrays = [avg for _ in range(y_train.shape[0])]
    AVG = np.stack(arrays, axis=0)
    index = np.isnan(y_train)
    y_train[index] = AVG[index]
    y_train[np.isnan(y_train)] = np.nanmean(y_train)
    return y_train


if __name__ == '__main__':
    data = pd.read_csv('/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/data/data_2016_I235E.csv',delimiter=',')
    _,Traffic = load_data(data)
    Traffic = np.swapaxes(Traffic,1,2)
    Raw_Speed = Traffic[:,:,:,0]
    Speed = convert_zero_2_mean(Raw_Speed)
    Speed = np.reshape(Speed, (Speed.shape[0],Speed.shape[1], Speed.shape[2]))

    print(Traffic.shape)
    print(Raw_Speed.shape)
    print(Speed.shape)

    # print(Speed[16,:5,:])
    # plt.pcolor(Speed[16,:,:],cmap=my_cmap)
    print(np.count_nonzero(Speed==0))
    print(np.count_nonzero(np.isnan(Speed)))

    data['y'][0][:4]
    data['day'] = data['y'].map(lambda x: x[:8])
    data['date'] = pd.to_datetime(data['day'],format='%Y%m%d')
    data['dayofweek'] = data['date'].map(lambda x: x.dayofweek)
    data['dayofyear'] = data['date'].map(lambda x: x.dayofyear)
    # del data['day']

    pickle.dump( (Traffic,Speed,data), open( "data.p", "wb" ) )
