import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *

from lib.Function_Smoothing import moving_avg_batch,data_smoothing_moving_avg

def get_certain_dayofweek(metadata, Speed, dayofweek = 0):
    data_sub = metadata[:len(Speed)]
    Mon = Speed[data_sub.index[data_sub['dayofweek'] == dayofweek],:,:]
    mon=np.mean(Mon,axis=0)
    mon_std=np.std(Mon,axis=0)
    Mon_delta = Mon - mon
    mon_std[mon_std<0.1] = np.mean(mon_std)
    Mon_Z = Mon_delta/mon_std
    return Mon, Mon_delta, Mon_Z, mon, mon_std

def MinMax_Normalization(samples):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0]*samples_shape[1]*samples_shape[2],1))
    scaler = MinMaxScaler().fit(samples)
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2]))
    return samples_normalized, scaler

def transfer_scale(samples,scaler):
    samples_shape = samples.shape
    samples = np.reshape(samples,(samples_shape[0]*samples_shape[1]*samples_shape[2],1))
    samples_normalized = scaler.transform(samples)
    samples_normalized = np.reshape(samples_normalized,(samples_shape[0],samples_shape[1],samples_shape[2]))
    return samples_normalized

def visualize_smooth(Speed):
    # visualize the speed data under different smoothing window
    Speed1 = data_smoothing_moving_avg(Speed,1)
    Speed5 = data_smoothing_moving_avg(Speed,5)
    Speed10 = data_smoothing_moving_avg(Speed,10)
    Speed30 = data_smoothing_moving_avg(Speed,30)

    gs = gridspec.GridSpec(4, 1, wspace=0, hspace=0.25)

    fig = plt.figure(figsize=(15,12))

    dayid=333
    sensorid=0
    ax = plt.subplot(gs[0])
    ax.set_title('12-19-2016 Monday non-smoothed')
    plt.plot(Speed1[dayid,:,sensorid].flatten())
    ax = plt.subplot(gs[1])
    ax.set_title('5min moving average')
    plt.plot(Speed5[dayid,:,sensorid].flatten())
    ax = plt.subplot(gs[2])
    ax.set_title('10min moving average')
    plt.plot(Speed10[dayid,:,sensorid].flatten())
    ax = plt.subplot(gs[3])
    ax.set_title('30min moving average')
    plt.plot(Speed30[dayid,:,sensorid].flatten())

    # fig.savefig('images/Speed_by_different_smoothing_window.png')

def visualize_avg(data,Speed):
    cdict = {'red': ((0.0, 1.0, 1.0),
                     (0.125, 1.0, 1.0),
                     (0.25, 1.0, 1.0),
                     (0.5625, 1.0, 1.0),
                     (1.0, 0.0, 0.0)),
            'green': ((0.0, 0.0, 0.0),
                     (0.25, 0.0, 0.0),
                     (0.5625, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
            'blue': ((0.0, 0.0, 0.0),
                     (0.5, 0.0, 0.0),
                     (1.0, 0.0, 0.0))}
    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    
    Mon, _, _, mon, _ = get_certain_dayofweek(data,Speed,0)
    Tue, _, _, tue, _ = get_certain_dayofweek(data,Speed,1)
    Wed, _, _, wed, _ = get_certain_dayofweek(data,Speed,2)
    Thu, _, _, thu, _ = get_certain_dayofweek(data,Speed,3)
    Fri, _, _, fri, _ = get_certain_dayofweek(data,Speed,4)
    Sat, _, _, sat, _ = get_certain_dayofweek(data,Speed,5)
    Sun, _, _, sun, _ = get_certain_dayofweek(data,Speed,6)
    
    gs = gridspec.GridSpec(7, 5, wspace=0, hspace=0.15)
    
    fig = plt.figure(figsize=(15,22))
    
    ax = plt.subplot(gs[0,0])
    plt.plot(mon[:,0])
    ax = plt.subplot(gs[0,1])
    plt.plot(mon[:,7])
    ax = plt.subplot(gs[0,2])
    plt.plot(mon[:,14])
    ax = plt.subplot(gs[0,3:])
    plt.pcolor(np.swapaxes(mon,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Monday')
    colorbar()
    ax = plt.subplot(gs[1,0])
    plt.plot(tue[:,0])
    ax = plt.subplot(gs[1,1])
    plt.plot(tue[:,7])
    ax = plt.subplot(gs[1,2])
    plt.plot(tue[:,14])
    ax = plt.subplot(gs[1,3:])
    plt.pcolor(np.swapaxes(tue,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Tuesday')
    colorbar()
    ax = plt.subplot(gs[2,0])
    plt.plot(wed[:,0])
    ax = plt.subplot(gs[2,1])
    plt.plot(wed[:,7])
    ax = plt.subplot(gs[2,2])
    plt.plot(wed[:,14])
    ax = plt.subplot(gs[2,3:])
    plt.pcolor(np.swapaxes(wed,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Wednesday')
    colorbar()
    ax = plt.subplot(gs[3,0])
    plt.plot(thu[:,0])
    ax = plt.subplot(gs[3,1])
    plt.plot(thu[:,7])
    ax = plt.subplot(gs[3,2])
    plt.plot(thu[:,14])
    ax = plt.subplot(gs[3,3:])
    plt.pcolor(np.swapaxes(thu,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Thursday')
    colorbar()
    ax = plt.subplot(gs[4,0])
    plt.plot(fri[:,0])
    ax = plt.subplot(gs[4,1])
    plt.plot(fri[:,7])
    ax = plt.subplot(gs[4,2])
    plt.plot(fri[:,14])
    ax = plt.subplot(gs[4,3:])
    plt.pcolor(np.swapaxes(fri,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Friday')
    colorbar()
    ax = plt.subplot(gs[5,0])
    plt.plot(sat[:,0])
    ax = plt.subplot(gs[5,1])
    plt.plot(sat[:,7])
    ax = plt.subplot(gs[5,2])
    plt.plot(sat[:,14])
    ax = plt.subplot(gs[5,3:])
    plt.pcolor(np.swapaxes(sat,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('Saturday')
    colorbar()
    ax = plt.subplot(gs[6,0])
    plt.plot(sun[:,0])
    ax = plt.subplot(gs[6,1])
    plt.plot(sun[:,7])
    ax = plt.subplot(gs[6,2])
    plt.plot(sun[:,14])
    ax = plt.subplot(gs[6,3:])
    plt.pcolor(np.swapaxes(sun,0,1),cmap=my_cmap, vmin=20, vmax=70)
    colorbar()
    ax.set_title('Sunday')
    
#    fig.savefig('AvgSpeed_by_dayofweek.png')


if __name__ == '__main__':
    (Traffic,Speed,data) = pickle.load( open( "/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/speed_short_term.p", "rb" ) )
    Traffic = data_smoothing_moving_avg(Traffic,5)
    Speed = data_smoothing_moving_avg(Speed,5)
    visualize_avg()

