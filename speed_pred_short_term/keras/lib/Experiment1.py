import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *

from keras.models import Sequential, Model, model_from_json
from keras.layers import concatenate, merge, Dense, LSTM, Input, Reshape, Convolution2D, Deconvolution2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.activations import relu, softmax, linear
from keras.layers.advanced_activations import PReLU, ELU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from lib.Function_DOW_Normalizer import get_certain_dayofweek,MinMax_Normalization,transfer_scale
from lib.Function_CreateInput import shapeback,create_dataset,create_dataset_historyAsFeature,create_dataset_historyAsSecondInput
from lib.Function_VisualizeResult import history_plot,history_plot_historyAsFeature,history_plot_historyAsSecondInput


def Experiment1(epochs,data_train,data_test,Speed):
#    select day of week for model development
    test_speed = Speed[data_test.index,:,:]
    train_speed = Speed[data_train.index,:,:]
    print('train_speed.shape = ',train_speed.shape)
    print('test_speed.shape = ',test_speed.shape)
    
#    construct model inputs
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
    
#    visualize test data
    #plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM
    plt.plot(test_speed_y[0,:,:])  #12-16-2016 Friday
    
#    one day is a batch
    batch_size = train_speed_x.shape[1]
    
#    define model structure
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False, return_sequences=True))
    # model.add(Dropout(0.3))
    model.add(LSTM(32, input_shape=(look_back, train_speed_x.shape[3]), stateful=False))
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
#    fit model
    train_x = np.reshape(train_speed_x,(train_speed_x.shape[0]*train_speed_x.shape[1],train_speed_x.shape[2],train_speed_x.shape[3]))
    train_y = np.reshape(train_speed_y,(train_speed_y.shape[0]*train_speed_y.shape[1],train_speed_y.shape[2]))
    history = model.fit(train_x, train_y,epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    return model,history

if __name__ == '__main__':
    epochs = 1000
    (Traffic,Speed,data) = pickle.load( open( "/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/speed_short_term.p", "rb" ) )
    Count = Traffic[:,:,:,1]
    Occup = Traffic[:,:,:,2]
    Traffic = data_smoothing_moving_avg(Traffic,5)
    Speed = data_smoothing_moving_avg(Speed,5)
    data_test = data.loc[[316,319,330],]
    data_train = data.loc[list(range(316))+list(range(317,319))+list(range(320,330))+list(range(331,len(data))),]
    model,history = Experiment1(epochs,data_train,data_test,Speed)
#    model.save_weights('images/weights/exp1.hdf5')
    history_plot(history,'images/history_exp1.png','images/test1_exp1.png',scoreflag=True,look_ahead = 1400,start = 0,test_case=1)
    history_plot(history,'images/history_exp1.png','images/test2_exp1.png',scoreflag=False,look_ahead = 1400,start = 0,test_case=2)
    history_plot(history,'images/history_exp1.png','images/test3_exp1.png',scoreflag=False,look_ahead = 1400,start = 0,test_case=3)
