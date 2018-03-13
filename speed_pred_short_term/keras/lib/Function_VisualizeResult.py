import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import *

from lib.Function_DOW_Normalizer import get_certain_dayofweek,MinMax_Normalization,transfer_scale
from lib.Function_CreateInput import create_dataset

(Traffic,Speed,data) = pickle.load( open( "/Users/Shuo/study/Project-predictive_study/Speed_Pred_Stage2/speed_short_term.p", "rb" ) )
train_speed, _, _, _, _ = get_certain_dayofweek(data,Speed[:334],dayofweek = 0)
test_speed = train_speed[-1:]

look_back = 15
mode = 'uni'
test_speed_x,test_speed_y = create_dataset(test_speed,test_speed, look_back, mode)

def model_score(history_object,image1):
    trainScore = [];
    for i in range(len(train_speed_x)):
        trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x)):
        testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')

def model_score2(history_object,image1):
    trainScore = [];
    for i in range(len(train_speed_x1)):
        trainScore.append(model.evaluate([train_speed_x1[i,:,:,:],train_speed_x2[i,:,:,:]], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x1)):
        testScore.append(model.evaluate([test_speed_x1[i,:,:,:],test_speed_x2[i,:,:,:]], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')

def history_plot(history_object,image1,image2,a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1)),scoreflag=True,look_ahead = 120,start = 390,test_case=1):
    
    if scoreflag:
        model_score(history_object,image1)
    
    fig1 = plt.figure(figsize=(12,20))
    pred_ranges = [1,5,10,15]
    subplot_id = 0

    for pred_range in pred_ranges:
        subplot_id += 1
        predictions = np.zeros((look_ahead,1))
        for i in range(look_ahead):
            trainPredict = test_speed_x[test_case-1,start+i,:,:]
            for j in range(pred_range):
                prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
                trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])
            predictions[i] = prediction
        
        ax = plt.subplot(len(pred_ranges),1,subplot_id)
        ax.set_title('{} min prediction'.format(pred_range), fontsize=20)
        
        plt.plot(np.arange(start+pred_range,start+look_ahead+pred_range),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
        plt.plot(np.arange(start+pred_range,start+look_ahead+pred_range),test_speed_y[test_case-1,(start+pred_range-1):(start+look_ahead+pred_range-1),:1]+a[(start+pred_range-1):(start+look_ahead+pred_range-1),:1],label="test function")
        plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

def history_plot_historyAsSecondInput(history_object,image1,image2,a=np.zeros((test_speed_y.shape[1],1)),b=np.zeros((test_speed_y.shape[1],1)),scoreflag=True,look_ahead = 120,start = 390,test_case=1):
    
    if scoreflag:
        model_score2(history_object,image1)
    
    fig1 = plt.figure(figsize=(12,20))
    pred_ranges = [1,5,10,15]
    subplot_id = 0

    for pred_range in pred_ranges:
        subplot_id += 1
        predictions = np.zeros((look_ahead,1))
        for i in range(look_ahead):
            trainPredict = test_speed_x1[test_case-1,start+i,:,:]
            input2 = test_speed_x2[test_case-1,start+i,:,:]
            for j in range(pred_range):
                prediction = model.predict([np.array([trainPredict]),np.array([input2])], batch_size=batch_size)
                trainPredict = np.vstack([trainPredict[1:],prediction+b[(start+i):(start+i+1),:1]])
            predictions[i] = prediction
        
        ax = plt.subplot(len(pred_ranges),1,subplot_id)
        ax.set_title('{} min prediction'.format(pred_range), fontsize=20)
        
        plt.plot(np.arange(start+pred_range,start+look_ahead+pred_range),predictions+a[start:(start+look_ahead),:1],'r',label="prediction")
        plt.plot(np.arange(start+pred_range,start+look_ahead+pred_range),test_speed_y[test_case-1,(start+pred_range-1):(start+look_ahead+pred_range-1),:1]+a[(start+pred_range-1):(start+look_ahead+pred_range-1),:1],label="test function")
        plt.legend()

    fig1.savefig(image2, bbox_inches='tight')

def history_plot_historyAsFeature(history_object,image1,image2):
    
    trainScore = [];
    for i in range(len(train_speed_x)):
        trainScore.append(model.evaluate(train_speed_x[i,:,:,:], train_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    trainScore = np.mean(trainScore)
    print('Train Score: ', trainScore)
    testScore = [];
    for i in range(len(test_speed_x)):
        testScore.append(model.evaluate(test_speed_x[i,:,:,:], test_speed_y[i,:,:], batch_size=batch_size, verbose=0))
    testScore = np.mean(testScore)
    print('Test Score: ', testScore)

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(history_object.history['loss'])
    plt.title('model mean squared error loss (Train Score:' + str(trainScore) + ' Test Score:' + str(testScore))
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    fig2.savefig(image1, bbox_inches='tight')
    
    look_ahead = 120
    start = 390
    trainPredict = test_speed_x[0,start,:,:]
    predictions = np.zeros((look_ahead,1))
    
    for i in range(look_ahead):
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction
        trainPredict = np.hstack([np.vstack([trainPredict[1:,:1],prediction]),test_speed_x[0,start+i+1,:,:1]])

    fig1 = plt.figure(figsize=(12,10))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title('prediction at the start of day', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),0],label="test function")
    plt.legend()

    predictions = np.zeros((look_ahead,1))
    
    for i in range(look_ahead):
        trainPredict = test_speed_x[0,start+i,:,:]
        prediction = model.predict(np.array([trainPredict]), batch_size=batch_size)
        predictions[i] = prediction


    ax2 = plt.subplot(2,1,2)
    ax2.set_title('prediction using real-time data', fontsize=20)
    plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
    plt.plot(np.arange(look_ahead),test_speed_y[0,start:(start+look_ahead),0],label="test function")
    plt.legend()

    fig1.savefig(image2, bbox_inches='tight')


if __name__ == '__main__':
    print('Hello')
