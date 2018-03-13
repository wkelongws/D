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

(Traffic,Speed,data) = pickle.load( open( "/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/speed_short_term.p", "rb" ) )


def Experiment2(epochs):
#    select day of week for model development
    train_speed = Speed[:334,:,:]
    print('train_speed.shape = ',train_speed.shape)

#    construct model inputs
    look_back = 15
    mode = 'uni'
    train_speed_x,train_speed_y = create_dataset_historyAsFeature(train_speed,train_speed, look_back, mode)
    # test_speed_x,test_speed_y = create_dataset_historyAsFeature(test_speed, look_back, mode)
    test_speed_x = train_speed_x[-1:,:,:,:]
    test_speed_y = train_speed_y[-1:,:,:]
    train_speed_x = train_speed_x[:-1,:,:,:]
    train_speed_y = train_speed_y[:-1,:,:]
    print('look_back = ',look_back)
    print('mode = ',mode)
    print('train_speed_x.shape = ',train_speed_x.shape)
    print('train_speed_y.shape = ',train_speed_y.shape)
    print('test_speed_x.shape = ',test_speed_x.shape)
    print('test_speed_y.shape = ',test_speed_y.shape)
    
#    visualize test data
    plt.plot(test_speed_y[0,390:510,:])  #12-19-2016 Monday 6:30AM - 8:30AM
    
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
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

    return history

if __name__ == '__main__':
    epochs = 1000
    history = Experiment2(epochs)
    history_plot_historyAsFeature(history,'images/history_exp2.png','images/test_exp2.png')
