import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
from scipy.stats import levy_stable

from datetime import datetime
from scipy.stats import kstest
from scipy.stats import jarque_bera
# from arch.unitroot import ADF
from scipy.stats import kurtosis
from scipy.stats import skew
# from arch import arch_model

import pickle

import ta

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

from scipy.signal import argrelmin
from scipy.signal import argrelmax

plt.style.use('classic')


import MetaTrader5 as mt
import pandas as pd
import ml_collections
import yaml

from datetime import datetime, timedelta
import pytz
import sys




def rsi_summary(df_observe, RSI_params, threshold, timeframe, direction_after_hit_threshold, show_distribution):
    df = df_observe.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['CLOSE'],
                                        window = RSI_params).rsi()
    
    df['GRP_RSI'] = pd.qcut(df['RSI'], 10)

    ## Smooth the timeline to determine number of candles that dissatisfy the threshold (3 candles)
    if timeframe == '1D':
        time_cont_dis_threshold = timedelta(days = -3)
        big_timeframe = 'MONTH'
    elif timeframe == '4H':
        time_cont_dis_threshold = timedelta(hours = -3*4)
        big_timeframe = 'WEEK'
    elif timeframe == '15min':
        time_cont_dis_threshold = timedelta(minutes = -3*15)
        big_timeframe = 'WEEK'
    elif timeframe == '1min':
        time_cont_dis_threshold = timedelta(minutes = -3*1)
        big_timeframe = 'WEEK'
    else:
        raise AssertionError('Input suitable timeframe (1min/ 15min/ 4H/ 1D).')

    if show_distribution:
        df_temp = pd.pivot_table(
            df, 
            index = [big_timeframe],
            columns = 'GRP_RSI',
            values = 'CLOSE',
            aggfunc = 'count',
            margins = True
        )

        df_temp = df_temp.apply(lambda x: x/x['All'], axis = 1)

        print('='*100)
        print(f'RSI distribution by {big_timeframe}')
        print('='*100)
        display(df_temp.style.background_gradient(axis = 0, cmap='RdYlGn_r'))
        



    print('='*100)
    print(f'RSI threshold reversion')
    print('='*100)
    if direction_after_hit_threshold == 'UP':
        df_threshold = df[df['RSI'] < threshold].copy()
    elif direction_after_hit_threshold == 'DOWN':
        df_threshold = df[df['RSI'] > threshold].copy()
    else:
        raise AssertionError('Input direction after hitting the threshold (UP/ DOWN).')

    df_threshold['DATE'] = df_threshold.index
    df_threshold['PREVIOUS_DIS_THRESHOLD_DATE'] = df_threshold['DATE'].shift(1)
    df_threshold['FLAG_CONTINUOUS_DIS_THRESHOLD'] = np.where(df_threshold['DATE'] + time_cont_dis_threshold < df_threshold['PREVIOUS_DIS_THRESHOLD_DATE'], 1, 0)

    dates_initial_dis_threshold = df_threshold[df_threshold['FLAG_CONTINUOUS_DIS_THRESHOLD'] == 0].index

    df_temp = pd.DataFrame(index = dates_initial_dis_threshold)

    for t in range(len(dates_initial_dis_threshold)):
        if t < len(dates_initial_dis_threshold) - 1:
            # Number of candles to continue to disrespect the threshold
            df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] = len(df_threshold[(df_threshold.index > dates_initial_dis_threshold[t]) \
                                                                                                    & (df_threshold.index < dates_initial_dis_threshold[t+1])])
            # Number of up/ down candles when the RSI continue to below 30
            df_temp.loc[dates_initial_dis_threshold[t], 'NUM_UP_CANDLE_DIS_THRESHOLD'] = df_threshold.loc[(df_threshold.index > dates_initial_dis_threshold[t]) \
                                                                                                    & (df_threshold.index < dates_initial_dis_threshold[t+1]), 'FLAG_INCREASE_CANDLE'].sum()
        else:
            # Same as above but different slicing for end of loop
            df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] = len(df_threshold[(df_threshold.index > dates_initial_dis_threshold[t])])
            
            df_temp.loc[dates_initial_dis_threshold[t], 'NUM_UP_CANDLE_DIS_THRESHOLD'] = df_threshold.loc[(df_threshold.index > dates_initial_dis_threshold[t]), 'FLAG_INCREASE_CANDLE'].sum()
            

    for t in range(len(dates_initial_dis_threshold)):
        ## Set the timeline to to observe 3, 7, 14 candles after satisfy the threshold
        if timeframe == '1D':
            start_time = timedelta(days = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'])
            end_time_1 = timedelta(days = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 4)
            end_time_2 = timedelta(days = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 8)    
            end_time_3 = timedelta(days = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 15)    
        elif timeframe == '4H':
            start_time = timedelta(hours = 4*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'])
            end_time_1 = timedelta(hours = 4*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 4*4)
            end_time_2 = timedelta(hours = 4*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 8*4)
            end_time_3 = timedelta(hours = 4*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 15*4)
        elif timeframe == '15min':
            start_time = timedelta(minutes = 15*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'])
            end_time_1 = timedelta(minutes = 15*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 4*15)
            end_time_2 = timedelta(minutes = 15*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 8*15)
            end_time_3 = timedelta(minutes = 15*df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 15*15)
        elif timeframe == '1min':
            start_time = timedelta(minutes = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'])
            end_time_1 = timedelta(minutes = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 4)
            end_time_2 = timedelta(minutes = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 8)
            end_time_3 = timedelta(minutes = df_temp.loc[dates_initial_dis_threshold[t], 'NUM_CANDLES_CONT_DIS_THRESHOLD'] + 15)


        df_temp.loc[dates_initial_dis_threshold[t], '3_CANDLES_NUM_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_1), 
                                                                                                    'FLAG_INCREASE_CANDLE'].sum()
        df_temp.loc[dates_initial_dis_threshold[t], '7_CANDLES_NUM_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_2), 
                                                                                                    'FLAG_INCREASE_CANDLE'].sum()
        df_temp.loc[dates_initial_dis_threshold[t], '14_CANDLES_NUM_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_3), 
                                                                                                    'FLAG_INCREASE_CANDLE'].sum()
        
        # Avg/ min/ max/ sum values of body candles - 7 candles
        df_temp.loc[dates_initial_dis_threshold[t], '7_CANDLES_AVG_BODY_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_2) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 1), 
                                                                                                    'BODY'].mean()
        df_temp.loc[dates_initial_dis_threshold[t], '7_CANDLES_AVG_BODY_DOWN_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_2) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 0), 
                                                                                                    'BODY'].mean()
        
        # Avg/ min/ max/ sum values of whole range candles
        df_temp.loc[dates_initial_dis_threshold[t], '7_CANDLES_AVG_WHOLE_RANGE_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_2) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 1), 
                                                                                                    'WHOLE_RANGE'].mean()
        df_temp.loc[dates_initial_dis_threshold[t], '7_CANDLES_AVG_WHOLE_RANGE_DOWN_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_2) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 0), 
                                                                                                    'WHOLE_RANGE'].mean()
        

        # Avg/ min/ max/ sum values of whole range candles - 14 candles
        df_temp.loc[dates_initial_dis_threshold[t], '14_CANDLES_AVG_WHOLE_RANGE_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_3) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 1), 
                                                                                                    'WHOLE_RANGE'].mean()
        df_temp.loc[dates_initial_dis_threshold[t], '14_CANDLES_AVG_WHOLE_RANGE_DOWN_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_3) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 0), 
                                                                                                    'WHOLE_RANGE'].mean()
        

        # Avg/ min/ max/ sum values of body candles - 14 candles
        df_temp.loc[dates_initial_dis_threshold[t], '14_CANDLES_AVG_BODY_UP_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_3) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 1), 
                                                                                                    'BODY'].mean()
        df_temp.loc[dates_initial_dis_threshold[t], '14_CANDLES_AVG_BODY_DOWN_CANDLE_THRESHOLD'] = df.loc[(df.index > dates_initial_dis_threshold[t] + start_time) \
                                                                                                    & (df.index <= dates_initial_dis_threshold[t] + end_time_3) \
                                                                                                    & (df['FLAG_INCREASE_CANDLE'] == 0), 
                                                                                                    'BODY'].mean()



    fig, ax = plt.subplots(nrows = 6, ncols = 1, figsize = (8, 15))

    ax[0].hist(df_temp['NUM_CANDLES_CONT_DIS_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'NUM_CANDLES_CONT_DIS_THRESHOLD')
    ax[0].hist(df_temp['NUM_UP_CANDLE_DIS_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'NUM_UP_CANDLE_DIS_THRESHOLD')
    ax[0].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    ax[1].hist(df_temp['7_CANDLES_NUM_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'NUM_UP_CANDLE_THRESHOLD(T+7)')
    ax[1].hist(df_temp['14_CANDLES_NUM_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'NUM_UP_CANDLE_THRESHOLD(T+14)')
    ax[1].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    ax[2].hist(df_temp['7_CANDLES_AVG_BODY_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'AVG_BODY_UP_THRESHOLD(T+7)')
    ax[2].hist(df_temp['7_CANDLES_AVG_BODY_DOWN_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'AVG_BODY_DOWN_THRESHOLD(T+7)')
    ax[2].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    ax[3].hist(df_temp['7_CANDLES_AVG_WHOLE_RANGE_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'AVG_WHOLE_RANGE_UP_THRESHOLD(T+7)')
    ax[3].hist(df_temp['7_CANDLES_AVG_WHOLE_RANGE_DOWN_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'AVG_WHOLE_RANGE_DOWN_THRESHOLD(T+7)')
    ax[3].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    ax[4].hist(df_temp['14_CANDLES_AVG_BODY_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'AVG_BODY_UP_THRESHOLD(T+14)')
    ax[4].hist(df_temp['14_CANDLES_AVG_BODY_DOWN_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'AVG_BODY_DOWN_THRESHOLD(T+14)')
    ax[4].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    ax[5].hist(df_temp['14_CANDLES_AVG_WHOLE_RANGE_UP_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'blue', label = 'AVG_WHOLE_RANGE_UP_THRESHOLD(T+14)')
    ax[5].hist(df_temp['14_CANDLES_AVG_WHOLE_RANGE_DOWN_CANDLE_THRESHOLD'], density = True, bins = 'auto', color = 'red', alpha = 0.8, label = 'AVG_WHOLE_RANGE_DOWN_THRESHOLD(T+14)')
    ax[5].legend(fontsize = 10, bbox_to_anchor = (1.6, 1))

    plt.show()

    # display(df_temp)



def ema_summary(df_observe, EMA_params, timeframe):
    df = df_observe.copy()

    for param in EMA_params:
        df[f'EMA({param})'] = ta.trend.EMAIndicator(df['CLOSE'],
                                            window = param).ema_indicator()
        df[f'POSITION_EMA({param})'] = df.apply(lambda x: 1 if x[f'EMA({param})'] >= x['HIGH']
                                                                    else (2 if x[f'EMA({param})'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x[f'EMA({param})'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x[f'EMA({param})'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)

    ## Smooth the timeline to determine number of candles that dissatisfy the threshold (3 candles)
    if timeframe == '1D':
        time_cont_below = timedelta(days = -3)
    elif timeframe == '4H':
        time_cont_below = timedelta(hours = -3*4)
    elif timeframe == '15min':
        time_cont_below = timedelta(minutes = -3*15)
    elif timeframe == '1min':
        time_cont_below = timedelta(minutes = -3*1)
    else:
        raise AssertionError('Input suitable timeframe (1min/ 15min/ 4H/ 1D).')

    lists_of_indicators = [f'EMA({col})' for col in EMA_params]
    lists_of_indicators = ['CLOSE', *lists_of_indicators]

    for idd, ind in enumerate(lists_of_indicators[0:-1]):
        for _, sub_ind in enumerate(lists_of_indicators[idd+1:]):
            print('='*100)
            print(f'{ind} and {sub_ind}')
            print('='*100)
            # Below the benchmark

            df_below = df[df[ind] < df[sub_ind]].copy()

            df_below['DATE'] = df_below.index
            df_below['PREVIOUS_BELOW_DATE'] = df_below['DATE'].shift(1)
            df_below['FLAG_CONTINUOUS_BELOW'] = np.where(df_below['DATE'] + time_cont_below < df_below['PREVIOUS_BELOW_DATE'], 1, 0)

            dates_initial_below = df_below[df_below['FLAG_CONTINUOUS_BELOW'] == 0].index

            df_temp_below = pd.DataFrame(index = dates_initial_below)

            for t in range(len(dates_initial_below)):
                if t < len(dates_initial_below) - 1:
                    # Number of candles to continue to disrespect the threshold
                    df_temp_below.loc[dates_initial_below[t], 'NUM_CANDLES_CONT_BELOW'] = len(df_below[(df_below.index > dates_initial_below[t]) \
                                                                                                            & (df_below.index < dates_initial_below[t+1])])
                    # Number of up/ down candles when the RSI continue to below 30
                    df_temp_below.loc[dates_initial_below[t], 'NUM_UP_CANDLE_BELOW'] = df_below.loc[(df_below.index > dates_initial_below[t]) \
                                                                                                            & (df_below.index < dates_initial_below[t+1]), 'FLAG_INCREASE_CANDLE'].sum()
                else:
                    # Same as above but different slicing for end of loop
                    df_temp_below.loc[dates_initial_below[t], 'NUM_CANDLES_CONT_BELOW'] = len(df_below[(df_below.index > dates_initial_below[t])])
                    
                    df_temp_below.loc[dates_initial_below[t], 'NUM_UP_CANDLE_BELOW'] = df_below.loc[(df_below.index > dates_initial_below[t]), 'FLAG_INCREASE_CANDLE'].sum()


            # Above the benchmark        

            time_cont_above = time_cont_below

            df_above = df[df[ind] > df[sub_ind]].copy()

            df_above['DATE'] = df_above.index
            df_above['PREVIOUS_ABOVE_DATE'] = df_above['DATE'].shift(1)
            df_above['FLAG_CONTINUOUS_ABOVE'] = np.where(df_above['DATE'] + time_cont_above < df_above['PREVIOUS_ABOVE_DATE'], 1, 0)

            dates_initial_above = df_above[df_above['FLAG_CONTINUOUS_ABOVE'] == 0].index

            df_temp_above = pd.DataFrame(index = dates_initial_above)

            for t in range(len(dates_initial_above)):
                if t < len(dates_initial_above) - 1:
                    # Number of candles to continue to disrespect the threshold
                    df_temp_above.loc[dates_initial_above[t], 'NUM_CANDLES_CONT_ABOVE'] = len(df_above[(df_above.index > dates_initial_above[t]) \
                                                                                                            & (df_above.index < dates_initial_above[t+1])])
                    # Number of up/ down candles when the RSI continue to below 30
                    df_temp_above.loc[dates_initial_above[t], 'NUM_UP_CANDLE_ABOVE'] = df_above.loc[(df_above.index > dates_initial_above[t]) \
                                                                                                            & (df_above.index < dates_initial_above[t+1]), 'FLAG_INCREASE_CANDLE'].sum()
                else:
                    # Same as above but different slicing for end of loop
                    df_temp_above.loc[dates_initial_above[t], 'NUM_CANDLES_CONT_ABOVE'] = len(df_above[(df_above.index > dates_initial_above[t])])
                    
                    df_temp_above.loc[dates_initial_above[t], 'NUM_UP_CANDLE_ABOVE'] = df_above.loc[(df_above.index > dates_initial_above[t]), 'FLAG_INCREASE_CANDLE'].sum()
                    

            df_temp_below['PERC_NUM_UP_CANDLE_BELOW'] = (df_temp_below['NUM_UP_CANDLE_BELOW']/df_temp_below['NUM_CANDLES_CONT_BELOW'])
            df_temp_above['PERC_NUM_UP_CANDLE_ABOVE'] = (df_temp_above['NUM_UP_CANDLE_ABOVE']/df_temp_above['NUM_CANDLES_CONT_ABOVE'])
            
            fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 8), sharex = False, sharey = False)

            ax[0, 0].hist(df_temp_below['NUM_CANDLES_CONT_BELOW'], density = True, bins = 'auto', label = 'NUM_CANDLES_CONT_BELOW')
            ax[0, 0].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))
            ax[0, 1].hist(df_temp_below['PERC_NUM_UP_CANDLE_BELOW'], density = True, bins = 'auto', color = 'green', label = 'NUM_UP_CANDLE_BELOW')
            ax[0, 1].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))

            ax[1, 0].hist(df_temp_above['NUM_CANDLES_CONT_ABOVE'], density = True, bins = 'auto', label = 'NUM_CANDLES_CONT_ABOVE')
            ax[1, 0].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))
            ax[1, 1].hist(df_temp_above['PERC_NUM_UP_CANDLE_ABOVE'], density = True, bins = 'auto', color = 'green', label = 'NUM_UP_CANDLE_ABOVE')
            ax[1, 1].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))
            
            plt.show()




def bollingerband_summary(df_observe, timeframe, mean, std):
    df = df_observe.copy()

    #Bollinger band
    df[f'BB_UPPER_BAND({mean})'] = ta.volatility.BollingerBands(df['CLOSE'], window = mean, window_dev = std).bollinger_hband()
    df[f'BB_LOWER_BAND({mean})'] = ta.volatility.BollingerBands(df['CLOSE'], window = mean, window_dev = std).bollinger_lband()
    df[f'BB_DIFF({mean})'] = df[f'BB_UPPER_BAND({mean})'] - df[f'BB_LOWER_BAND({mean})']
    df[f'FLAG_BB_EXPAND({mean}, 20)'] = np.where(df[f'BB_DIFF({mean})'] > df[f'BB_DIFF({mean})'].shift(20), 1, 0)

    ## Smooth the timeline to determine number of candles that dissatisfy the threshold (3 candles)
    if timeframe == '1D':
        time_cont_below = timedelta(days = -3)
    elif timeframe == '4H':
        time_cont_below = timedelta(hours = -3*4)
    elif timeframe == '15min':
        time_cont_below = timedelta(minutes = -3*15)
    elif timeframe == '1min':
        time_cont_below = timedelta(minutes = -3*1)
    else:
        raise AssertionError('Input suitable timeframe (1min/ 15min/ 4H/ 1D).')

    lists_of_indicators = [f'BB_UPPER_BAND({mean})', f'BB_LOWER_BAND({mean})']

    for idd, ind in enumerate(lists_of_indicators):

        print('='*100)
        print(f'{ind}')
        print('='*100)
        # Below the benchmark

        df_below = df[df['CLOSE'] < df[ind]].copy()

        df_below['DATE'] = df_below.index
        df_below['PREVIOUS_BELOW_DATE'] = df_below['DATE'].shift(1)
        df_below['FLAG_CONTINUOUS_BELOW'] = np.where(df_below['DATE'] + time_cont_below < df_below['PREVIOUS_BELOW_DATE'], 1, 0)

        dates_initial_below = df_below[df_below['FLAG_CONTINUOUS_BELOW'] == 0].index

        df_temp_below = pd.DataFrame(index = dates_initial_below)

        for t in range(len(dates_initial_below)):
            if t < len(dates_initial_below) - 1:
                # Number of candles to continue to disrespect the threshold
                df_temp_below.loc[dates_initial_below[t], 'NUM_CANDLES_CONT_BELOW'] = len(df_below[(df_below.index > dates_initial_below[t]) \
                                                                                                        & (df_below.index < dates_initial_below[t+1])])
                # Number of up/ down candles when the RSI continue to below 30
                df_temp_below.loc[dates_initial_below[t], 'NUM_UP_CANDLE_BELOW'] = df_below.loc[(df_below.index > dates_initial_below[t]) \
                                                                                                        & (df_below.index < dates_initial_below[t+1]), 'FLAG_INCREASE_CANDLE'].sum()
            else:
                # Same as above but different slicing for end of loop
                df_temp_below.loc[dates_initial_below[t], 'NUM_CANDLES_CONT_BELOW'] = len(df_below[(df_below.index > dates_initial_below[t])])
                
                df_temp_below.loc[dates_initial_below[t], 'NUM_UP_CANDLE_BELOW'] = df_below.loc[(df_below.index > dates_initial_below[t]), 'FLAG_INCREASE_CANDLE'].sum()


        # Above the benchmark        

        time_cont_above = time_cont_below

        df_above = df[df['CLOSE'] > df[ind]].copy()

        df_above['DATE'] = df_above.index
        df_above['PREVIOUS_ABOVE_DATE'] = df_above['DATE'].shift(1)
        df_above['FLAG_CONTINUOUS_ABOVE'] = np.where(df_above['DATE'] + time_cont_above < df_above['PREVIOUS_ABOVE_DATE'], 1, 0)

        dates_initial_above = df_above[df_above['FLAG_CONTINUOUS_ABOVE'] == 0].index

        df_temp_above = pd.DataFrame(index = dates_initial_above)

        for t in range(len(dates_initial_above)):
            if t < len(dates_initial_above) - 1:
                # Number of candles to continue to disrespect the threshold
                df_temp_above.loc[dates_initial_above[t], 'NUM_CANDLES_CONT_ABOVE'] = len(df_above[(df_above.index > dates_initial_above[t]) \
                                                                                                        & (df_above.index < dates_initial_above[t+1])])
                # Number of up/ down candles when the RSI continue to below 30
                df_temp_above.loc[dates_initial_above[t], 'NUM_UP_CANDLE_ABOVE'] = df_above.loc[(df_above.index > dates_initial_above[t]) \
                                                                                                        & (df_above.index < dates_initial_above[t+1]), 'FLAG_INCREASE_CANDLE'].sum()
            else:
                # Same as above but different slicing for end of loop
                df_temp_above.loc[dates_initial_above[t], 'NUM_CANDLES_CONT_ABOVE'] = len(df_above[(df_above.index > dates_initial_above[t])])
                
                df_temp_above.loc[dates_initial_above[t], 'NUM_UP_CANDLE_ABOVE'] = df_above.loc[(df_above.index > dates_initial_above[t]), 'FLAG_INCREASE_CANDLE'].sum()
                

        df_temp_below['PERC_NUM_UP_CANDLE_BELOW'] = (df_temp_below['NUM_UP_CANDLE_BELOW']/df_temp_below['NUM_CANDLES_CONT_BELOW'])
        df_temp_above['PERC_NUM_UP_CANDLE_ABOVE'] = (df_temp_above['NUM_UP_CANDLE_ABOVE']/df_temp_above['NUM_CANDLES_CONT_ABOVE'])
        
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 8), sharex = False, sharey = False)

        ax[0, 0].hist(df_temp_below['NUM_CANDLES_CONT_BELOW'], density = True, bins = 'auto', label = 'NUM_CANDLES_CONT_BELOW')
        ax[0, 0].legend(fontsize = 10, bbox_to_anchor = (0.5, 1))
        ax[0, 1].hist(df_temp_below['PERC_NUM_UP_CANDLE_BELOW'], density = True, bins = 'auto', color = 'green', label = 'NUM_UP_CANDLE_BELOW')
        ax[0, 1].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))

        ax[1, 0].hist(df_temp_above['NUM_CANDLES_CONT_ABOVE'], density = True, bins = 'auto', label = 'NUM_CANDLES_CONT_ABOVE')
        ax[1, 0].legend(fontsize = 10, bbox_to_anchor = (0.5, 1))
        ax[1, 1].hist(df_temp_above['PERC_NUM_UP_CANDLE_ABOVE'], density = True, bins = 'auto', color = 'green', label = 'NUM_UP_CANDLE_ABOVE')
        ax[1, 1].legend(fontsize = 10, bbox_to_anchor = (1.35, 1))
        
        plt.show()






class Backtest_report:
    def __init__(self, alpha, 
                 df_is,
                 base_SL = 10, base_TP = 20, max_existing_positions = 3, init_vol = 0.01, incre_vol = 0.01, max_vol = 0.1,
                 init_cap = 1000, incre_cap = 2, asset = 'XAUUSD', re_allocation = True
                 ):
        self.alpha = alpha # alpha class
        self.df_is = df_is # data
        self.init_vol = init_vol # initial volume
        self.re_allocation = re_allocation
        self.incre_vol = incre_vol # increment of volume
        self.max_vol = max_vol # max value of volume
        self.init_cap = init_cap # init capital
        self.incre_cap = incre_cap # the condition of capital if increasing the trading volume
        self.asset = asset
        self.max_existing_positions = max_existing_positions
        self.base_SL = base_SL
        self.base_TP = base_TP    
        self.df_result_is = None
        self.df_balance_is = None
        self.df_balance_new = None
        
        
    def _cal_balance(self, df_balance):
        df_balance['CUMULATIVE_PNL'] = df_balance['PNL'].cumsum()
        df_balance['BALANCE'] = df_balance['CUMULATIVE_PNL'] + 1000

        # Adjust the balance, if hit 0 then set anything afterward = 0
        df_balance['BALANCE'] = df_balance['BALANCE'].apply(lambda x: x if x >= 0 else 0)
        if len(df_balance[df_balance['BALANCE'] == 0]) != 0:
            df_balance.loc[df_balance.index >= df_balance[df_balance['BALANCE'] == 0].index[0], 'BALANCE'] = 0
            df_balance['PNL'] = np.where(df_balance['BALANCE'] == 0, 0, df_balance['PNL'])
            df_balance['CUMULATIVE_PNL'] = np.where(df_balance['BALANCE'] == 0, 0, df_balance['CUMULATIVE_PNL'])
        
        df_balance['DRAWDOWN'] = np.where(df_balance['PNL'] < 0, df_balance['PNL']/(df_balance['BALANCE'] - df_balance['PNL']), 0)
        df_balance['CUMULATIVE_DRAWDOWN'] = np.where(df_balance['PNL'] < 0, df_balance['CUMULATIVE_PNL']/(df_balance['BALANCE'] - df_balance['CUMULATIVE_PNL']), 0)

        return(df_balance)


    def prepare_report(self):
        df_signal_is = self.alpha.signal(self.df_is, eval = True)
        df_result_is = self.df_is.merge(df_signal_is, how = 'left', on = 'DATE_TIME')
        df_result = df_result_is
                
        signals = df_result[df_result['SIGNAL'] != 0]

        for ids in range(len(signals)):
            s = signals.iloc[ids, :]
            df_temp = df_result.loc[(df_result.index > s.name) 
                    & (((df_result['CLOSE'] - s.CLOSE)*s.SIGNAL < s.SL*(-1)) | ((df_result['CLOSE'] - s.CLOSE)*s.SIGNAL > s.TP)), 
                    :]
            if len(df_temp) != 0:
                df_result.loc[s.name, 'TIME_CLOSE_POSITION'] = df_temp.index[0]
            else:
                df_result.loc[s.name, 'TIME_CLOSE_POSITION'] = None

        signals = df_result[df_result['SIGNAL'] != 0]
        df_result['VOL'] = self.init_vol
        df_result['FLAG_VALID_POSITION'] = 1


        if self.max_existing_positions != None:
            existing_positions = signals[:self.max_existing_positions]

            for ids in range(len(signals)):
                if ids >= self.max_existing_positions:
                    s = signals.iloc[ids, :]

                    if s.name <= existing_positions['TIME_CLOSE_POSITION'].min():
                        df_result.loc[s.name, 'FLAG_VALID_POSITION'] = 0
                    else: 
                        df_result.loc[s.name, 'FLAG_VALID_POSITION'] = 1
                        existing_positions = existing_positions.iloc[1:, :]
                        existing_positions = pd.concat([existing_positions, pd.DataFrame(s).transpose()], axis = 0)

            df_result['FLAG_VALID_POSITION'] = df_result['FLAG_VALID_POSITION'].fillna(1)


        df_result = df_result.merge(df_result[['CLOSE']], 
                                                how = 'left', 
                                                left_on = 'TIME_CLOSE_POSITION', 
                                                right_index = True,
                                                suffixes = ('_open', '_close')
                                                )

        df_result['PNL'] = (df_result['CLOSE_close'] - df_result['CLOSE_open'])*df_result['SIGNAL']*df_result['FLAG_VALID_POSITION']*(self.init_vol*100)
        df_result['PNL'] = np.where(df_result['PNL'] < self.base_SL*-1*(self.init_vol*100), self.base_SL*-1*(self.init_vol*100), 
                                    np.where(df_result['PNL'] > self.base_TP*(self.init_vol*100), self.base_TP*(self.init_vol*100), 
                                            np.where(df_result['PNL'].isnull() == True, 0, df_result['PNL'])))

        # Calculate balance
        df_count = df_result.loc[(df_result['SIGNAL'] != 0) & (df_result['FLAG_VALID_POSITION'] == 1), 
                                 ['TIME_CLOSE_POSITION', 'PNL']].dropna().sort_values(by = 'TIME_CLOSE_POSITION')
        df_count = pd.pivot_table(
            df_count,
            index = 'TIME_CLOSE_POSITION',
            values = 'PNL',
            aggfunc = 'count'
        )
        
        df_balance = df_result[['TIME_CLOSE_POSITION', 'PNL']].dropna().sort_values(by = 'TIME_CLOSE_POSITION')
        df_balance = pd.pivot_table(
            df_balance,
            index = 'TIME_CLOSE_POSITION',
            values = 'PNL',
            aggfunc = 'sum'
        )

        df_balance = df_balance.merge(df_count, how = 'left', on = 'TIME_CLOSE_POSITION')
        df_balance.columns = ['PNL', 'COUNT']
        df_balance['VOL'] = self.init_vol
        df_balance = self._cal_balance(df_balance)
        
        self.df_result_is = df_result[df_result.index <= df_result_is.index[-1]] 
        self.df_balance_is = df_balance[df_balance.index <= df_result_is.index[-1]] 

        if self.re_allocation:
            idt = self.df_balance_is.index[0]

        df_balance_new = self.df_balance_is.copy()
        cap = self.init_cap
        vol = self.init_vol
        
        for idd, d in enumerate(df_balance_new.index): 
            if df_balance_new.loc[d, 'BALANCE']/cap >= self.incre_cap:
                idt =  d
                cap *= self.incre_cap
                vol += self.incre_vol

                if vol >= self.max_vol:
                    vol = self.max_vol

                df_balance_new.loc[df_balance_new.index > idt, 'PNL'] *= vol*100
                df_balance_new.loc[df_balance_new.index > idt, 'VOL'] = vol
                df_balance_new = self._cal_balance(df_balance_new[['PNL', 'COUNT', 'VOL']].copy())
                
        self.df_balance_new = df_balance_new

    
    def display_report(self):
        if self.df_result_is == None and self.df_balance_is == None:
            self.prepare_report()

        if self.re_allocation:
            size = (35, 30)
        else:
            size = (35, 20)

        fig = plt.figure(figsize = size)

        ax = fig.add_subplot(5, 1, 1)
        ax.plot(self.df_balance_is.loc[self.df_balance_is['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')

        ax_0 = ax.twinx()
        ax_0.plot(self.df_balance_is.loc[self.df_balance_is['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
        ax_0.hlines(y = -0.05, xmin = self.df_balance_is.loc[self.df_balance_is['BALANCE'] > 0, :].index[0], 
                    xmax = self.df_balance_is.loc[self.df_balance_is['BALANCE'] > 0, :].index[-1], 
                    color='r', linestyles = '--', alpha = 0.3)
        
        ax.legend(bbox_to_anchor = (1.1, 1))
        ax_0.legend(bbox_to_anchor = (1.1, 0.8))
        ax.set_title('Original Balance and Drawdown (No re-allocation)')
        
        
        df_summary = pd.concat(
            [
                pd.pivot_table(
                    self.df_result_is[(self.df_result_is['FLAG_VALID_POSITION'] == 1) & (self.df_result_is['SIGNAL'] != 0) & (self.df_result_is['PNL'] > 0)],
                    index = 'SIGNAL',
                    values = 'CLOSE_open',
                    aggfunc = 'count',
                    margins = True
                ),
                pd.pivot_table(
                    self.df_result_is[(self.df_result_is['FLAG_VALID_POSITION'] == 1) & (self.df_result_is['SIGNAL'] != 0) & (self.df_result_is['PNL'] < 0)],
                    index = 'SIGNAL',
                    values = 'CLOSE_open',
                    aggfunc = 'count',
                    margins = True
                )
            ],
            axis = 1
        )

        df_summary.columns = ['WINNING_POSITIONS', 'LOSING_POSITIONS']
        df_summary.index = ['SHORT', 'LONG', 'ALL']
        df_summary = df_summary.fillna(0)
        df_summary['TOTAL_POSITIONS'] = df_summary['WINNING_POSITIONS'] + df_summary['LOSING_POSITIONS']
        
        df_summary['WINNING_POSITIONS'] = df_summary['WINNING_POSITIONS']/df_summary['TOTAL_POSITIONS']
        df_summary['LOSING_POSITIONS'] = df_summary['LOSING_POSITIONS']/df_summary['TOTAL_POSITIONS']
        df_summary['TOTAL_POSITIONS'] = df_summary['TOTAL_POSITIONS']/df_summary['TOTAL_POSITIONS']
        
        ax = fig.add_subplot(5, 1, 2)
        ax.bar(df_summary.index, df_summary['WINNING_POSITIONS'], color = 'green', label = 'WINNING_POSITIONS')
        ax.bar(df_summary.index, df_summary['LOSING_POSITIONS'], bottom = df_summary['WINNING_POSITIONS'], color = 'red', label = 'LOSING_POSITIONS')
        
        ax.legend()
        ax.set_title('Distribution of winning/ losing positions')

        ax = fig.add_subplot(5, 1, 3)
        ax.plot(self.df_balance_is.loc[self.df_balance_is['BALANCE'] > 0, 'COUNT'], color = 'blue', label = 'NUM_POSITIONS')
        ax.legend(bbox_to_anchor = (1.1, 1))        
        ax.set_title('Number of positions')

        if self.re_allocation:
            ax = fig.add_subplot(5, 1, 4)

            ax.plot(self.df_balance_new.loc[self.df_balance_new['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')

            ax_0 = ax.twinx()
            ax_0.plot(self.df_balance_new.loc[self.df_balance_new['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
            ax_0.hlines(y = -0.05, xmin = self.df_balance_new.loc[self.df_balance_new['BALANCE'] > 0, :].index[0], 
                        xmax = self.df_balance_new.loc[self.df_balance_new['BALANCE'] > 0, :].index[-1], 
                        color='r', linestyles = '--', alpha = 0.3)
            
            ax.legend(bbox_to_anchor = (1.1, 1))
            ax_0.legend(bbox_to_anchor = (1.1, 0.8))    
            ax.set_title('New Balance and Drawdown (with re-allocation)')
          
            ax = fig.add_subplot(5, 1, 5)
            ax.plot(self.df_balance_new.loc[self.df_balance_new['BALANCE'] > 0, 'VOL'], color = 'blue', label = 'VOL')
            ax.legend(bbox_to_anchor = (1.1, 1))
            ax.set_ylim((0, 0.1))
            ax.set_title('Volume')


    

        plt.show()
