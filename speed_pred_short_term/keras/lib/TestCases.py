import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *

from lib.Function_DOW_Normalizer import get_certain_dayofweek

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

#Mon, _, _, mon, _ = get_certain_dayofweek(data,Speed,0)
#Tue, _, _, tue, _ = get_certain_dayofweek(data,Speed,1)
#Wed, _, _, wed, _ = get_certain_dayofweek(data,Speed,2)
#Thu, _, _, thu, _ = get_certain_dayofweek(data,Speed,3)
#Fri, _, _, fri, _ = get_certain_dayofweek(data,Speed,4)
#Sat, _, _, sat, _ = get_certain_dayofweek(data,Speed,5)
#Sun, _, _, sun, _ = get_certain_dayofweek(data,Speed,6)

def test_case_1(data,Speed):
    # test case 1: recurrent morning congestion
    Mon, _, _, mon, _ = get_certain_dayofweek(data,Speed,0)
    
    fig1=plt.figure(figsize=(15,8))
    fig1.suptitle('test case 1 (recurrent morning congestion): 12-05-2016, Monday')
    ax=plt.subplot(2,2,1)
    plt.pcolor(np.swapaxes(mon,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax.set_title('average traffic speed on Mondays')
    ax=plt.subplot(2,2,2)
    plt.pcolor(np.swapaxes(Speed[319,:,:],0,1),cmap=my_cmap, vmin=20, vmax=70) #12-05-2016 Monday 6:30AM - 8:30AM
    ax.set_title('traffic speed on the target Monday')
    ax=plt.subplot(2,2,3)
    ax.set_title('average traffic speed of sensor0 on Mondays')
    plt.plot(mon[:,0])
    ax.set_ylim([0,70])
    ax=plt.subplot(2,2,4)
    ax.set_title('traffic speed of sensor0 on the target Monday')
    plt.plot(Speed[319,:,0])  #12-05-2016 Monday 6:30AM - 8:30AM sensor 1
    ax.set_ylim([0,70])

    return fig1

def test_case_2(data,Speed):
    # test case 2: recurrent morning non-congestion
    Fri, _, _, fri, _ = get_certain_dayofweek(data,Speed,4)
    
    fig2=plt.figure(figsize=(15,8))
    fig2.suptitle('test case 2 (recurrent morning non-congestion): 12-02-2016, Friday')
    ax=plt.subplot(2,2,1)
    ax.set_title('average traffic speed on Fridays')
    plt.pcolor(np.swapaxes(fri,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax=plt.subplot(2,2,2)
    ax.set_title('average traffic speed on the target Friday')
    plt.pcolor(np.swapaxes(Speed[316,:,:],0,1),cmap=my_cmap, vmin=20, vmax=70)  #12-02-2016 Friday 6:30AM - 8:30AM
    ax=plt.subplot(2,2,3)
    ax.set_title('average traffic speed of sensor0 on Fridays')
    plt.plot(fri[:,0])
    ax.set_ylim([0,70])
    ax=plt.subplot(2,2,4)
    ax.set_title('traffic speed of sensor0 on the target Friday')
    plt.plot(Speed[316,:,0])  #12-02-2016 Friday 6:30AM - 8:30AM
    ax.set_ylim([0,70])

    return fig2

def test_case_3(data,Speed):
    # test case 3: non-recurrent morning congestion
    Fri, _, _, fri, _ = get_certain_dayofweek(data,Speed,4)
    
    fig3=plt.figure(figsize=(15,8))
    fig3.suptitle('test case 3 (non-recurrent morning congestion): 12-16-2016, Friday')
    ax=plt.subplot(2,2,1)
    ax.set_title('average traffic speed on Fridays')
    plt.pcolor(np.swapaxes(fri,0,1),cmap=my_cmap, vmin=20, vmax=70)
    ax=plt.subplot(2,2,2)
    ax.set_title('average traffic speed on the target Friday')
    plt.pcolor(np.swapaxes(Speed[330,:,:],0,1),cmap=my_cmap, vmin=20, vmax=70)  #12-16-2016 Friday 6:30AM - 8:30AM
    ax=plt.subplot(2,2,3)
    ax.set_title('average traffic speed of sensor0 on Fridays')
    plt.plot(fri[:,0])
    ax.set_ylim([0,70])
    ax=plt.subplot(2,2,4)
    ax.set_title('traffic speed of sensor0 on the target Friday')
    plt.plot(Speed[330,:,0])  #12-16-2016 Friday 6:30AM - 8:30AM
    ax.set_ylim([0,70])

    return fig3

if __name__ == '__main__':

    print('啥也没有')
