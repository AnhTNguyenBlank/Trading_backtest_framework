import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
from datetime import datetime

import ta

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

import ml_collections
import yaml

plt.style.use('classic')

import MetaTrader5 as mt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import contextlib
import os


def prepare_df(df, timeframe, add_indicators):

    assert timeframe in ['1min', '5min', '15min', '4h', '1D']

    if timeframe != '1min':
        df = df.resample(rule = timeframe).agg(
            {'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'CLOSE': 'last',
            'TICK_VOL': 'sum',
            }).dropna()

    df['AVG_PRICE'] = (df['OPEN'] + df['HIGH'] + df['LOW'] + df['CLOSE'])/4

    df['FLAG_INCREASE_CANDLE'] = np.where(df['CLOSE'] > df['OPEN'], 1, 0)    
    
    df['BODY'] = df.apply(lambda x: max(x['OPEN'], x['CLOSE']) - min(x['OPEN'], x['CLOSE']),
                                    axis = 1)
    df['UPPER_SHADOW'] = df.apply(lambda x: x['HIGH'] - max(x['OPEN'], x['CLOSE']),
                                            axis = 1)
    df['LOWER_SHADOW'] = df.apply(lambda x: min(x['OPEN'], x['CLOSE']) - x['LOW'],
                                            axis = 1)
    df['WHOLE_RANGE'] = df['HIGH'] - df['LOW']

    df['FLAG_LONG_UPPER_SHADOW'] = np.where(df['UPPER_SHADOW'] >= df['BODY'], 1, 0)
    df['FLAG_LONG_LOWER_SHADOW'] = np.where(df['LOWER_SHADOW'] >= df['BODY'], 1, 0)

    df['FLAG_HIGHER_HIGH(20)'] = np.where(df['HIGH'] >= df['HIGH'].shift(20), 1, 0)
    df['FLAG_HIGHER_LOW(20)'] = np.where(df['LOW'] >= df['LOW'].shift(20), 1, 0)


    #Moving average of TICK_VOL
    df['AVG_VOL(50)'] = df['TICK_VOL'].rolling(50).mean()
    df['FLAG_OVER_AVG_VOL(50)'] = np.where(df['TICK_VOL'] >= df['AVG_VOL(50)'], 1, 0)

    df['AVG_VOL(200)'] = df['TICK_VOL'].rolling(200).mean()
    df['FLAG_OVER_AVG_VOL(200)'] = np.where(df['TICK_VOL'] >= df['AVG_VOL(200)'], 1, 0)

    df['FLAG_UPTREND_VOL(20)'] = np.where(df['TICK_VOL'] >= df['TICK_VOL'].shift(20), 1, 0)


    if add_indicators:
        #RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['CLOSE'],
                                                window = 7).rsi()

        df['FLAG_UNDER_30_RSI'] = np.where(df['RSI'] < 30, 1, 0)
        df['FLAG_OVER_70_RSI'] = np.where(df['RSI'] > 70, 1, 0)
        df['FLAG_UPTREND_RSI(20)'] = np.where(df['RSI'] >= df['RSI'].shift(20), 1, 0)
        
        #Bollinger band
        df['BB_UPPER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_hband()
        df['POSITION_UPPER_BAND(50)'] = df.apply(lambda x: 1 if x['BB_UPPER_BAND(50)'] >= x['HIGH']
                                                                    else (2 if x['BB_UPPER_BAND(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['BB_UPPER_BAND(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['BB_UPPER_BAND(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)
        
        df['BB_LOWER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_lband()
        df['POSITION_LOWER_BAND(50)'] = df.apply(lambda x: 1 if x['BB_LOWER_BAND(50)'] >= x['HIGH']
                                                                    else (2 if x['BB_LOWER_BAND(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['BB_LOWER_BAND(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['BB_LOWER_BAND(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)
        
        
        #Exponential moving average
        df['EMA(50)'] = ta.trend.EMAIndicator(df['CLOSE'],
                                                window = 50).ema_indicator()
        df['POSITION_EMA(50)'] = df.apply(lambda x: 1 if x['EMA(50)'] >= x['HIGH']
                                                                    else (2 if x['EMA(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['EMA(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['EMA(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)
        

        df['EMA(200)'] = ta.trend.EMAIndicator(df['CLOSE'],
                                                window = 200).ema_indicator()
        df['POSITION_EMA(200)'] = df.apply(lambda x: 1 if x['EMA(200)'] >= x['HIGH']
                                                                    else (2 if x['EMA(200)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['EMA(200)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['EMA(200)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)

    #returns
    # df['Ret(t)'] = 100*(df['CLOSE'] - df['CLOSE'].shift(1))/df['CLOSE'].shift(1)

    return(df)


def get_session(hour):
    if 0 <= hour < 7:
        return 1
    elif 7 <= hour < 13:
        return 2
    elif 13 <= hour < 21:
        return 3
    else:
        return 4
    

def plot_df(df, path, open_tab):

    # Assuming df is your DataFrame with columns: 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL', 'EMA50', 'EMA200', 'RSI'

    # Define subplot heights and widths
    subplot_heights = [600, 100, 100]  # Adjust these values based on your preferences
    subplot_widths = [1]  # Only one column

    # Create subplot with 3 rows and 1 column
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Main Chart', 'TICK_VOL Chart', 'RSI Chart'),
                        row_heights=subplot_heights,
                        column_widths=subplot_widths,
                        vertical_spacing = 0.05,  # Set the spacing between rows
                        )

    # Subplot 1: Candlestick chart with EMA lines
    cd = go.Candlestick(x=df.index,
                    open=df['OPEN'],
                    high=df['HIGH'],
                    low=df['LOW'],
                    close=df['CLOSE'],
                    increasing=dict(line=dict(color='white', width = 2)),  # Adjust the line attributes for increasing candles
                    decreasing=dict(line=dict(color='blue', width = 2)),
                    name='Candlesticks',
                    opacity = 0.5
                    
                    )

    ema50 = go.Scatter(x=df.index,
                             y=df['EMA(50)'],
                             mode='lines',
                             name='EMA50',
                             line=dict(color='red', width = 2))

    ema200 = go.Scatter(x=df.index,
                             y=df['EMA(200)'],
                             mode='lines',
                             name='EMA200',
                             line=dict(color='yellow', width = 2))
    
    bb_upper = go.Scatter(x=df.index,
                             y=df['BB_UPPER_BAND(50)'],
                             mode='lines',
                             name='BB_UPPER_BAND50',
                             line=dict(color='white', width = 1))

    bb_lower = go.Scatter(x=df.index,
                             y=df['BB_LOWER_BAND(50)'],
                             mode='lines',
                             name='BB_LOWER_BAND50',
                             line=dict(color='white', width = 1))

    fig.add_trace(cd, row=1, col=1)
    fig.add_trace(ema50,
                  row=1,
                  col=1)
    fig.add_trace(ema200,
                  row=1,
                  col=1)
    fig.add_trace(bb_upper,
                  row=1,
                  col=1)
    fig.add_trace(bb_lower,
                  row=1,
                  col=1)
    

    # Subplot 2: TICK_VOL bar chart
    vol = go.Bar(x=df.index,
                         y=df['TICK_VOL'],
                         name='TICK_VOL',
                         marker=dict(color='blue'),
                         #width = 0
                         )

    av50 = go.Scatter(x=df.index,
                             y=df['AVG_VOL(50)'],
                             mode='lines',
                             name='AVG_VOL50',
                             line=dict(color='red', width = 2))

    av200 = go.Scatter(x=df.index,
                             y=df['AVG_VOL(200)'],
                             mode='lines',
                             name='AVG_VOL200',
                             line=dict(color='yellow', width = 2))


    fig.add_trace(vol,
                  row=2,
                  col=1)
    fig.add_trace(av50,
                  row=2,
                  col=1)
    fig.add_trace(av200,
                  row=2,
                  col=1)

    # Subplot 3: RSI chart with threshold lines

    rsi = go.Scatter(x=df.index,
                             y=df['RSI'],
                             mode='lines',
                             name='RSI',
                             line=dict(color='mediumpurple', width = 2))

    rsi30 = dict(type='line',
                       x0=df.index.min(),
                       x1=df.index.max(),
                       y0=30,
                       y1=30,
                       line=dict(color='white', width=1, dash='dash'))

    rsi70 = dict(type='line',
                       x0=df.index.min(),
                       x1=df.index.max(),
                       y0=70,
                       y1=70,
                       line=dict(color='white', width=1, dash='dash'))

    fig.add_trace(rsi,
                  row=3,
                  col=1)

    fig.add_shape(rsi30,
                  row=3,
                  col=1)

    fig.add_shape(rsi70, row=3, col=1)


    # Add darker shaded area between| 30 and 70 in the RSI plot
    fig.add_shape(
        type='rect',
        x0=df.index.min(),
        x1=df.index.max(),
        y0=30,
        y1=70,
        fillcolor='rgba(200, 160, 255, 0.2)',  # Light purple color with opacity
        line=dict(color='rgba(255, 255, 255, 0)'),  # Set border color and opacity
        row=3,
        col=1
    )

    # Add slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=False,
                thickness=0.05,  # Adjust the thickness of the slider
                bgcolor='rgba(0,0,0,0.1)',  # Set the background color of the slider
            ),
            ),

        height = 800,
        width = 1300,
        plot_bgcolor='black',  # Transparent background
        paper_bgcolor='black',  # Transparent paper background
        font = dict(color = 'white'),
        legend = dict(x = 1.01, y = 1),

        xaxis3_rangeslider_visible = True,
        xaxis3_rangeslider_thickness = 0.05,
        
    )

    # Fix y-axis range for each subplot
    fig.update_yaxes(autorange = True, 
                     # range=[df['CLOSE'].min(), df['CLOSE'].max()], 
                     row=1, col=1, fixedrange= False)  # Adjust as needed
    fig.update_yaxes(autorange = True, 
                     # range=[0, df['TICK_VOL'].max()], 
                     row=2, col=1, fixedrange= False)  # Adjust as needed
    fig.update_yaxes(autorange = True, 
                     range=[0, 100], 
                     row=3, col=1, 
                     # fixedrange= False
                    )  # Assuming RSI values range from 0 to 100


    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey',
        # rangebreaks=[
        #     dict(bounds=["sat", "mon"]),  # Exclude weekends
        #     # dict(bounds=[15, 9], pattern="hour"),  # hide hours outside of 9:00 - 15:00
        #     # dict(bounds=[12, 13], pattern="hour"),  # hide hours outside of 12:00 - 13:00
        # ]
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey'
    )

    if path:
        # Write HTML output
        fig.write_html(path)
        url = path
        if open_tab:
            webbrowser.open(url, new=2)  # open in new tab

    return(fig)


def wait_for_exact_second(target_second=0):
    while True:
        now = datetime.now()
        if now.second == target_second:
            break
        time.sleep(0.2)  # small sleep to avoid busy waiting
    print(f"Started at: {now.strftime('%Y-%m-%d %H:%M:%S')}")


def error_correcting(series, errors, alpha_1, alpha_2):
    return (
        alpha_1 * series
        + (1 - alpha_1) * errors.ewm(alpha=alpha_2).mean()
    ).astype("float64")


def apply_error_correction(df, src_cols, err_cols, out_cols, alpha_1=0.8, alpha_2=0.8):
    return pd.DataFrame(
        {
            out: error_correcting(df[src], df[err], alpha_1, alpha_2)
            for src, err, out in zip(src_cols, err_cols, out_cols)
        },
        index=df.index
    )


def finalize_data(
        ohlc_1m,
        ohlc_15m,
        ohlc_H4,
        ohlc_D1,
        scale_ohlc = True,
        scale_multiplier = 100
        ):

    if scale_ohlc:
        ohlc_1m[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = ohlc_1m[['OPEN', 'HIGH', 'LOW', 'CLOSE']]*scale_multiplier
        ohlc_15m[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = ohlc_15m[['OPEN', 'HIGH', 'LOW', 'CLOSE']]*scale_multiplier
        ohlc_H4[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = ohlc_H4[['OPEN', 'HIGH', 'LOW', 'CLOSE']]*scale_multiplier
        ohlc_D1[['OPEN', 'HIGH', 'LOW', 'CLOSE']] = ohlc_D1[['OPEN', 'HIGH', 'LOW', 'CLOSE']]*scale_multiplier

    ohlc_1m['KEY_MAP_15M'] = ohlc_1m.index.floor('15min') - pd.Timedelta(minutes = 15)
    ohlc_1m['KEY_MAP_4H'] = ohlc_1m.index.floor('4h') - pd.Timedelta(hours = 4)
    ohlc_1m['KEY_MAP_1D'] = ohlc_1m.index.date - pd.Timedelta(days = 1)

    ohlc_1m['KEY_MAP_15M'] = np.where(ohlc_1m['KEY_MAP_15M'].dt.day_of_week == 6, ohlc_1m['KEY_MAP_15M'] - pd.Timedelta(days = 2), ohlc_1m['KEY_MAP_15M'])
    ohlc_1m['KEY_MAP_4H'] = np.where(ohlc_1m['KEY_MAP_4H'].dt.day_of_week == 6, ohlc_1m['KEY_MAP_4H'] - pd.Timedelta(days = 2), ohlc_1m['KEY_MAP_4H'])
    ohlc_1m['KEY_MAP_1D'] = np.where(pd.to_datetime(ohlc_1m['KEY_MAP_1D']).dt.day_of_week == 6, pd.to_datetime(ohlc_1m['KEY_MAP_1D']) - pd.Timedelta(days = 2), pd.to_datetime(ohlc_1m['KEY_MAP_1D']))

    df_15_min_NE = pd.DataFrame()

    for time in list(ohlc_1m['KEY_MAP_15M'].unique()):
        df_temp = ohlc_1m[ohlc_1m['KEY_MAP_15M'] == time].copy()
        df_temp['OPEN_15min_NE'] = df_temp['OPEN'].values[0].copy()
        df_temp['HIGH_15min_NE'] = df_temp['HIGH'].cummax()
        df_temp['LOW_15min_NE'] = df_temp['LOW'].cummin()
        df_temp['CLOSE_15min_NE'] = df_temp['CLOSE'].copy()
        df_temp['TICK_VOL_15min_NE'] = df_temp['TICK_VOL'].cumsum()
        
        df_15_min_NE = pd.concat([df_15_min_NE, df_temp], axis = 0)

    df_15_min_NE = df_15_min_NE[['OPEN_15min_NE', 'HIGH_15min_NE', 'LOW_15min_NE', 'CLOSE_15min_NE', 'TICK_VOL_15min_NE']]
    df_15_min_NE.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL']
    #============================
    df_4_hour_NE = pd.DataFrame()

    for time in list(ohlc_1m['KEY_MAP_4H'].unique()):
        df_temp = ohlc_1m[ohlc_1m['KEY_MAP_4H'] == time].copy()
        df_temp['OPEN_4hour_NE'] = df_temp['OPEN'].values[0].copy()
        df_temp['HIGH_4hour_NE'] = df_temp['HIGH'].cummax()
        df_temp['LOW_4hour_NE'] = df_temp['LOW'].cummin()
        df_temp['CLOSE_4hour_NE'] = df_temp['CLOSE'].copy()
        df_temp['TICK_VOL_4hour_NE'] = df_temp['TICK_VOL'].cumsum()
        
        df_4_hour_NE = pd.concat([df_4_hour_NE, df_temp], axis = 0)

    df_4_hour_NE = df_4_hour_NE[['OPEN_4hour_NE', 'HIGH_4hour_NE', 'LOW_4hour_NE', 'CLOSE_4hour_NE', 'TICK_VOL_4hour_NE']]
    df_4_hour_NE.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL']
    #============================
    df_1_day_NE = pd.DataFrame()

    for time in list(ohlc_1m['KEY_MAP_1D'].unique()):
        df_temp = ohlc_1m[ohlc_1m['KEY_MAP_1D'] == time].copy()
        df_temp['OPEN_1day_NE'] = df_temp['OPEN'].values[0].copy()
        df_temp['HIGH_1day_NE'] = df_temp['HIGH'].cummax()
        df_temp['LOW_1day_NE'] = df_temp['LOW'].cummin()
        df_temp['CLOSE_1day_NE'] = df_temp['CLOSE'].copy()
        df_temp['TICK_VOL_1day_NE'] = df_temp['TICK_VOL'].cumsum()
        
        df_1_day_NE = pd.concat([df_1_day_NE, df_temp], axis = 0)

    df_1_day_NE = df_1_day_NE[['OPEN_1day_NE', 'HIGH_1day_NE', 'LOW_1day_NE', 'CLOSE_1day_NE', 'TICK_VOL_1day_NE']]
    df_1_day_NE.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL']

    df_1_min = prepare_df(df = ohlc_1m.copy(), timeframe = '1min', add_indicators = True)
    df_15_min = prepare_df(df = ohlc_15m.copy(), timeframe = '15min', add_indicators = True)
    df_4_hour = prepare_df(df = ohlc_H4.copy(), timeframe = '4h', add_indicators = True)
    df_1_day = prepare_df(df = ohlc_D1.copy(), timeframe = '1D', add_indicators = True)

    df_15_min_NE = prepare_df(df = df_15_min_NE.copy(), timeframe = '1min', add_indicators = True)
    df_4_hour_NE = prepare_df(df = df_4_hour_NE.copy(), timeframe = '1min', add_indicators = True)
    df_1_day_NE = prepare_df(df = df_1_day_NE.copy(), timeframe = '1min', add_indicators = True)

    # Readjust the parameter for indicators

    for col in df_1_min.columns:
        if col not in ['KEY_MAP_15M', 'KEY_MAP_4H', 'KEY_MAP_1D']:
            df_1_min[col] = df_1_min[col].astype('float64')

    for col in df_15_min.columns:
        df_15_min[col] = df_15_min[col].astype('float64')

    for col in df_4_hour.columns:
        df_4_hour[col] = df_4_hour[col].astype('float64')

    for col in df_1_day.columns:
        df_1_day[col] = df_1_day[col].astype('float64')

    for col in df_15_min_NE.columns:
        df_15_min_NE[col] = df_15_min_NE[col].astype('float64')

    for col in df_4_hour_NE.columns:
        df_4_hour_NE[col] = df_4_hour_NE[col].astype('float64')

    for col in df_1_day_NE.columns:
        df_1_day_NE[col] = df_1_day_NE[col].astype('float64')

    df_1_min['HOURS'] = df_1_min.index.hour
    df_1_min['SESSION'] = df_1_min['HOURS'].apply(get_session)

    df_15_min['HOURS'] = df_15_min.index.hour
    df_15_min['SESSION'] = df_15_min['HOURS'].apply(get_session)

    df_4_hour['HOURS'] = df_4_hour.index.hour
    df_4_hour['SESSION'] = df_4_hour['HOURS'].apply(get_session)

    df_15_min_NE['HOURS'] = df_15_min_NE.index.hour
    df_15_min_NE['SESSION'] = df_15_min_NE['HOURS'].apply(get_session)

    df_4_hour_NE['HOURS'] = df_4_hour_NE.index.hour
    df_4_hour_NE['SESSION'] = df_4_hour_NE['HOURS'].apply(get_session)
    
    df_1_min = df_1_min.dropna()
    df_15_min = df_15_min.dropna()
    df_4_hour = df_4_hour.dropna()
    df_1_day = df_1_day.dropna()

    df_15_min_NE = df_15_min_NE.dropna()
    df_4_hour_NE = df_4_hour_NE.dropna()
    df_1_day_NE = df_1_day_NE.dropna()

    normalizing_cols = [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'AVG_PRICE', 
        'BODY', 'UPPER_SHADOW', 'LOWER_SHADOW', 'WHOLE_RANGE',
        'TICK_VOL', 'AVG_VOL(50)', 'AVG_VOL(200)',
        'BB_UPPER_BAND(50)', 'BB_LOWER_BAND(50)',
        'EMA(50)', 'EMA(200)'
    ]

    non_normalizing_cols = [
        'FLAG_INCREASE_CANDLE',
        'FLAG_LONG_UPPER_SHADOW', 'FLAG_LONG_LOWER_SHADOW', 'FLAG_HIGHER_HIGH(20)', 'FLAG_HIGHER_LOW(20)',
        'FLAG_OVER_AVG_VOL(50)', 'FLAG_OVER_AVG_VOL(200)', 'FLAG_UPTREND_VOL(20)', 
        'RSI', 'FLAG_UNDER_30_RSI', 'FLAG_OVER_70_RSI', 'FLAG_UPTREND_RSI(20)',
        'POSITION_UPPER_BAND(50)', 'POSITION_LOWER_BAND(50)', 
        'POSITION_EMA(50)', 'POSITION_EMA(200)', 'Ret(t)', 'HOURS', 'SESSION'
        ]

    normalized_cols = ['NORM_' + col for col in normalizing_cols]

    timeframe = ['_1min', '_15min', '_4hour', '_1day', '_15min_NE', '_4hour_NE', '_1day_NE']

    # Normalized predictors

    df_1_min = df_1_min.copy()
    df_15_min = df_15_min.copy()
    df_4_hour = df_4_hour.copy()
    df_1_day = df_1_day.copy()

    df_15_min_NE = df_15_min_NE.copy()
    df_4_hour_NE = df_4_hour_NE.copy()
    df_1_day_NE = df_1_day_NE.copy()

    for col in normalized_cols:
        df_1_min[col] = ((df_1_min[col[5:]] - df_1_min[col[5:]].shift(1).rolling(20).mean())/df_1_min[col[5:]].shift(1).rolling(20).std()).astype('float64')
        df_15_min[col] = ((df_15_min[col[5:]] - df_15_min[col[5:]].shift(1).rolling(20).mean())/df_15_min[col[5:]].shift(1).rolling(20).std()).astype('float64')
        df_4_hour[col] = ((df_4_hour[col[5:]] - df_4_hour[col[5:]].shift(1).rolling(20).mean())/df_4_hour[col[5:]].shift(1).rolling(20).std()).astype('float64')
        df_1_day[col] = ((df_1_day[col[5:]] - df_1_day[col[5:]].shift(1).rolling(20).mean())/df_1_day[col[5:]].shift(1).rolling(20).std()).astype('float64')
        
        df_15_min_NE[col] = ((df_15_min_NE[col[5:]] - df_15_min_NE[col[5:]].shift(1).rolling(20).mean())/df_15_min_NE[col[5:]].shift(1).rolling(20).std()).astype('float64')
        df_4_hour_NE[col] = ((df_4_hour_NE[col[5:]] - df_4_hour_NE[col[5:]].shift(1).rolling(20).mean())/df_4_hour_NE[col[5:]].shift(1).rolling(20).std()).astype('float64')
        df_1_day_NE[col] = ((df_1_day_NE[col[5:]] - df_1_day_NE[col[5:]].shift(1).rolling(20).mean())/df_1_day_NE[col[5:]].shift(1).rolling(20).std()).astype('float64')

    # Sequential predictors    
    
    pct_cols = [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL',
        'AVG_PRICE', 'BODY',
        'UPPER_SHADOW', 'LOWER_SHADOW', 'WHOLE_RANGE', 
        'AVG_VOL(50)', 'AVG_VOL(200)',
        'RSI', 
        'BB_UPPER_BAND(50)', 'BB_LOWER_BAND(50)',
        'EMA(50)', 'EMA(200)',
        # 'NORM_OPEN', 'NORM_HIGH', 'NORM_LOW',
        # 'NORM_CLOSE', 'NORM_AVG_PRICE', 'NORM_BODY', 'NORM_UPPER_SHADOW',
        # 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_TICK_VOL',
        # 'NORM_AVG_VOL(50)', 'NORM_AVG_VOL(200)', 'NORM_BB_UPPER_BAND(50)',
        # 'NORM_BB_LOWER_BAND(50)', 'NORM_EMA(50)', 'NORM_EMA(200)'
    ]

    diff_pct_cols = ['DIFF_' + col for col in pct_cols]

    df_1_min = df_1_min.copy()
    df_15_min = df_15_min.copy()
    df_4_hour = df_4_hour.copy()
    df_1_day = df_1_day.copy()

    df_15_min_NE = df_15_min_NE.copy()
    df_4_hour_NE = df_4_hour_NE.copy()
    df_1_day_NE = df_1_day_NE.copy()

    for col in diff_pct_cols:
        df_1_min[col + '_(1)'] = ((df_1_min[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_1_min[col + '_(2)'] = ((df_1_min[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_1_min[col + '_(3)'] = (df_1_min[col[5:]] + 10e-1).pct_change(periods = 3)
        
        df_15_min[col + '_(1)'] = ((df_15_min[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_15_min[col + '_(2)'] = ((df_15_min[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_15_min[col + '_(3)'] = ((df_15_min[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')
        
        df_4_hour[col + '_(1)'] = ((df_4_hour[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_4_hour[col + '_(2)'] = ((df_4_hour[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_4_hour[col + '_(3)'] = ((df_4_hour[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')

        df_1_day[col + '_(1)'] = ((df_1_day[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_1_day[col + '_(2)'] = ((df_1_day[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_1_day[col + '_(3)'] = ((df_1_day[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')

        df_15_min_NE[col + '_(1)'] = ((df_15_min_NE[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_15_min_NE[col + '_(2)'] = ((df_15_min_NE[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_15_min_NE[col + '_(3)'] = ((df_15_min_NE[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')
        
        df_4_hour_NE[col + '_(1)'] = ((df_4_hour_NE[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_4_hour_NE[col + '_(2)'] = ((df_4_hour_NE[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_4_hour_NE[col + '_(3)'] = ((df_4_hour_NE[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')

        df_1_day_NE[col + '_(1)'] = ((df_1_day_NE[col[5:]] + 10e-1).pct_change(periods = 1)).astype('float64')
        df_1_day_NE[col + '_(2)'] = ((df_1_day_NE[col[5:]] + 10e-1).pct_change(periods = 2)).astype('float64')
        df_1_day_NE[col + '_(3)'] = ((df_1_day_NE[col[5:]] + 10e-1).pct_change(periods = 3)).astype('float64')
            
    diff_cols = [
        'FLAG_INCREASE_CANDLE', 
        'FLAG_LONG_UPPER_SHADOW',
        'FLAG_LONG_LOWER_SHADOW', 'FLAG_HIGHER_HIGH(20)', 'FLAG_HIGHER_LOW(20)',
        'FLAG_OVER_AVG_VOL(50)', 
        'FLAG_OVER_AVG_VOL(200)', 
        'FLAG_UPTREND_VOL(20)',
        'FLAG_UNDER_30_RSI', 'FLAG_OVER_70_RSI', 'FLAG_UPTREND_RSI(20)',
        'POSITION_UPPER_BAND(50)', 
        'POSITION_LOWER_BAND(50)', 
        'POSITION_EMA(50)', 
        'POSITION_EMA(200)', 
    ]

    diff_abs_cols = ['DIFF_' + col for col in diff_cols]

    df_1_min = df_1_min.copy()
    df_15_min = df_15_min.copy()
    df_4_hour = df_4_hour.copy()
    df_1_day = df_1_day.copy()

    df_15_min_NE = df_15_min_NE.copy()
    df_4_hour_NE = df_4_hour_NE.copy()
    df_1_day_NE = df_1_day_NE.copy()

    for col in diff_abs_cols:
        df_1_min[col + '_(1)'] = df_1_min[col[5:]].diff(periods = 1)
        df_1_min[col + '_(2)'] = df_1_min[col[5:]].diff(periods = 2)
        df_1_min[col + '_(3)'] = df_1_min[col[5:]].diff(periods = 3)

        df_15_min[col + '_(1)'] = df_15_min[col[5:]].diff(periods = 1)
        df_15_min[col + '_(2)'] = df_15_min[col[5:]].diff(periods = 2)
        df_15_min[col + '_(3)'] = df_15_min[col[5:]].diff(periods = 3)
        
        df_4_hour[col + '_(1)'] = df_4_hour[col[5:]].diff(periods = 1)
        df_4_hour[col + '_(2)'] = df_4_hour[col[5:]].diff(periods = 2)
        df_4_hour[col + '_(3)'] = df_4_hour[col[5:]].diff(periods = 3)
        
        df_1_day[col + '_(1)'] = df_1_day[col[5:]].diff(periods = 1)
        df_1_day[col + '_(2)'] = df_1_day[col[5:]].diff(periods = 2)
        df_1_day[col + '_(3)'] = df_1_day[col[5:]].diff(periods = 3)
        
        df_15_min_NE[col + '_(1)'] = df_15_min_NE[col[5:]].diff(periods = 1)
        df_15_min_NE[col + '_(2)'] = df_15_min_NE[col[5:]].diff(periods = 2)
        df_15_min_NE[col + '_(3)'] = df_15_min_NE[col[5:]].diff(periods = 3)
        
        df_4_hour_NE[col + '_(1)'] = df_4_hour_NE[col[5:]].diff(periods = 1)
        df_4_hour_NE[col + '_(2)'] = df_4_hour_NE[col[5:]].diff(periods = 2)
        df_4_hour_NE[col + '_(3)'] = df_4_hour_NE[col[5:]].diff(periods = 3)
        
        df_1_day_NE[col + '_(1)'] = df_1_day_NE[col[5:]].diff(periods = 1)
        df_1_day_NE[col + '_(2)'] = df_1_day_NE[col[5:]].diff(periods = 2)
        df_1_day_NE[col + '_(3)'] = df_1_day_NE[col[5:]].diff(periods = 3)
        
    # Finalizing df
    df_15_min = df_15_min.loc[:, ~df_15_min.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]
    df_15_min_NE = df_15_min_NE.loc[:, ~df_15_min_NE.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]

    df_4_hour = df_4_hour.loc[:, ~df_4_hour.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]
    df_4_hour_NE = df_4_hour_NE.loc[:, ~df_4_hour_NE.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]

    df_1_day = df_1_day.loc[:, ~df_1_day.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]
    df_1_day_NE = df_1_day_NE.loc[:, ~df_1_day_NE.columns.isin(['NORM_UPPER_SHADOW', 'NORM_LOWER_SHADOW', 'NORM_WHOLE_RANGE', 'NORM_HIGH', 'NORM_LOW', 'NORM_OPEN'])]

    useless_cols = [
        # 'OPEN', 'HIGH', 'LOW', 'CLOSE', 
        'TICK_VOL', 
        'SPREAD', 'REAL_VOLUME', 'FLAG_CANDLE_END', 
        'AVG_PRICE', 
        'AVG_VOL(50)', 
        'AVG_VOL(200)',
        'BB_UPPER_BAND(50)', 
        'BB_LOWER_BAND(50)',
        'EMA(50)', 
        'EMA(200)',
        'HOURS', 'SESSION', 
        'KEY_MAP_15M', 'KEY_MAP_4H', 'KEY_MAP_1D', 
        'POSITION_IN_15min', 'POSITION_IN_4hour', 'POSITION_IN_1day'
    ]

    useful_cols = [col for col in df_1_min.columns if col not in useless_cols]

    df_1_min.columns = [col + '_1min' for col in df_1_min.columns]
    df_15_min.columns = [col + '_15min' for col in df_15_min.columns]
    df_4_hour.columns = [col + '_4hour' for col in df_4_hour.columns]
    df_1_day.columns = [col + '_1day' for col in df_1_day.columns]

    df_15_min_NE.columns = [col + '_15min' for col in df_15_min_NE.columns]
    df_4_hour_NE.columns = [col + '_4hour' for col in df_4_hour_NE.columns]
    df_1_day_NE.columns = [col + '_1day' for col in df_1_day_NE.columns]

    useful_cols_1min = [col + '_1min' for col in useful_cols]
    useful_cols_15min = [col + '_15min' for col in useful_cols]
    useful_cols_4hour = [col + '_4hour' for col in useful_cols]
    useful_cols_1day = [col + '_1day' for col in useful_cols]

    useful_cols_15min_NE = [col + '_15min_NE' for col in useful_cols]
    useful_cols_4hour_NE = [col + '_4hour_NE' for col in useful_cols]
    useful_cols_1day_NE = [col + '_1day_NE' for col in useful_cols]

    useful_cols_15min = [col for col in useful_cols_15min if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols_15min_NE = [col for col in useful_cols_15min_NE if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols_4hour = [col for col in useful_cols_4hour if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols_4hour_NE = [col for col in useful_cols_4hour_NE if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols_1day = [col for col in useful_cols_1day if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols_1day_NE = [col for col in useful_cols_1day_NE if not ((col.startswith('NORM_OPEN')) \
                                            | (col.startswith('NORM_UPPER_SHADOW')) \
                                            | (col.startswith('NORM_LOWER_SHADOW')) \
                                            | (col.startswith('NORM_WHOLE_RANGE')) \
                                            | (col.startswith('NORM_HIGH')) \
                                            | (col.startswith('NORM_LOW')) \
                                            )]

    useful_cols = [*useful_cols_1min, *useful_cols_15min, *useful_cols_4hour, *useful_cols_1day]
    df_1_min = df_1_min.reset_index()
    df_15_min = df_15_min.reset_index()
    df_4_hour = df_4_hour.reset_index()
    df_1_day = df_1_day.reset_index()


    df = pd.merge_asof(df_1_min[useful_cols_1min + ['TIME', 'KEY_MAP_15M_1min', 'KEY_MAP_4H_1min', 'KEY_MAP_1D_1min']].copy(), 
                df_15_min[useful_cols_15min + ['TIME']].copy(), 
                left_on = 'KEY_MAP_15M_1min',
                right_on = 'TIME',
                suffixes = ('_1min', '_15min'),
                direction = 'backward',
                allow_exact_matches = True, 
                tolerance = pd.Timedelta("15m"),
                )

    df = pd.merge_asof(df.copy(), df_4_hour[useful_cols_4hour + ['TIME']].copy(), 
                left_on = 'KEY_MAP_4H_1min', 
                right_on = 'TIME', 
                direction = 'backward',
                allow_exact_matches = True, 
                tolerance = pd.Timedelta("4h"),
                )

    df['KEY_MAP_1D_1min'] = pd.to_datetime(df['KEY_MAP_1D_1min'])

    df = pd.merge_asof(df.copy(), df_1_day[useful_cols_1day + ['TIME']].copy(), 
                left_on = 'KEY_MAP_1D_1min', 
                right_on = 'TIME', 
                suffixes = ('_4hour', '_1day'),
                direction = 'backward',
                allow_exact_matches = True, 
                tolerance = pd.Timedelta("1D"),
                )

    df = df.drop(columns = ['TIME_15min', 'TIME_4hour', 'TIME_1day', 'KEY_MAP_15M_1min', 'KEY_MAP_4H_1min', 'KEY_MAP_1D_1min'])
    df = df.set_index('TIME_1min')
    df.index.name = 'DATE_TIME'

    df['POSITION_IN_15min'] = df.index.minute % 15 + 1
    df['POSITION_IN_4hour'] = (df.index.hour * 60 + df.index.minute) % 240 + 1
    df['POSITION_IN_1day'] = (df.index.hour * 60 + df.index.minute) % 1440 + 1

    df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.fillna(-999)
    df = df.dropna()


    # Error correcting
    id_drop = []

    for idx in df_15_min_NE.index:
        if idx not in df.index:
            id_drop.append(idx)

    df_15_min_NE = df_15_min_NE[~df_15_min_NE.index.isin(id_drop)]
    df_4_hour_NE = df_4_hour_NE[~df_4_hour_NE.index.isin(id_drop)]
    df_1_day_NE = df_1_day_NE[~df_1_day_NE.index.isin(id_drop)]

    df['FIRST_HOUR_AFTER_OPEN'] = np.where(df['POSITION_IN_1day'] <= 60, 1, 0).astype('int32')
    df['LAST_HOUR_BEFORE_CLOSE'] = np.where(df['POSITION_IN_1day'] >= 1381, 1, 0).astype('int32')

    df = pd.merge(
        df.copy(),
        df_15_min_NE.copy(),
        left_index = True,
        right_index = True,
        how = 'left',
        suffixes = ('', '_NE')
        )

    df = pd.merge(
        df.copy(),
        df_4_hour_NE.copy(),
        left_index = True,
        right_index = True,
        how = 'left',
        suffixes = ('', '_NE')
        )

    df = pd.merge(
        df.copy(),
        df_1_day_NE.copy(),
        left_index = True,
        right_index = True,
        how = 'left',
        suffixes = ('', '_NE')
        )

    useful_cols_15min_COR = [col + '_COR' for col in useful_cols_15min]
    useful_cols_4hour_COR = [col + '_COR' for col in useful_cols_4hour]
    useful_cols_1day_COR = [col + '_COR' for col in useful_cols_1day]

    df = df.copy()

    new_features = []
    new_features.append(
        apply_error_correction(
            df,
            useful_cols_15min,
            useful_cols_15min_NE,
            useful_cols_15min_COR,
        )
    )
    new_features.append(
        apply_error_correction(
            df,
            useful_cols_4hour,
            useful_cols_4hour_NE,
            useful_cols_4hour_COR,
        )
    )
    new_features.append(
        apply_error_correction(
            df,
            useful_cols_1day,
            useful_cols_1day_NE,
            useful_cols_1day_COR,
        )
    )

    df = pd.concat([df] + new_features, axis=1)

    int32_cols = df.select_dtypes(include='int32').columns
    df[int32_cols] = df[int32_cols].astype('int64')

    useful_cols = [*useful_cols_1min, 
        *useful_cols_15min, *useful_cols_15min_NE, *useful_cols_15min_COR,
        *useful_cols_4hour, *useful_cols_4hour_NE, *useful_cols_4hour_COR,
        *useful_cols_1day, *useful_cols_1day_NE, *useful_cols_1day_COR,
    ]

    return(df, useful_cols)

# =================================== Web scraping data (news) support =================================== #

def set_up_driver(num_clicks, time_sleep_open):
    '''
    This function only supports the scraping from this site: "https://www.businesstoday.in/news".
    It may support other sites but hadnot been tested on.
    Includes progress bar.
    '''
    # Setup headless Chrome
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")  # Only FATAL
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Redirect stderr (to hide native logs from Chrome/TensorFlow/C++)
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        driver = webdriver.Chrome(options=options)

    # Open the news page
    driver.get("https://www.businesstoday.in/news")
    time.sleep(time_sleep_open)  # Allow JS to load

    # Click the "Load More" button multiple times
    for _ in tqdm(range(num_clicks), desc="Loading more articles", unit = 'page'):  # Adjust range for more clicks
        load_more_button = driver.find_element(By.ID, "load_more")
        driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
        driver.execute_script("arguments[0].click();", load_more_button)
        
        # Wait until the spinner disappears, no matter how long it takes
        WebDriverWait(driver, timeout=60).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "circular_loader_container"))
        )

    return(driver)


def extract_article_content(url):
    '''
    This function only supports the scraping from this site: "https://www.businesstoday.in/news".
    It may support other sites but hadnot been tested on.
    '''
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the posted time of the article

    user_section = soup.find('div', class_='userdetail_share_main')
    if not user_section:
        return{"time": None, "content": "❌ Content section not found."}

    li_tag = user_section.find('li')
    if not li_tag:
        return{"time": None, "content": "❌ Content section not found."}
    

    raw_time = li_tag.get_text(strip=True)
    time_str = raw_time.replace("Updated", "").replace("IST", "").replace(",", "").strip()
    
    try:
        dt_naive = datetime.strptime(time_str, "%b %d %Y %I:%M %p")
        IST = timezone(timedelta(hours=5, minutes=30))
        GMT7 = timezone(timedelta(hours=7))
        dt_ist = dt_naive.replace(tzinfo=IST)
        dt_gmt7 = dt_ist.astimezone(GMT7)
    except ValueError:
        dt_naive = None
        dt_gmt7 = None

    
    # Extract the main content of the page
    main_div = soup.find('div', class_='story_witha_main_sec')
    if not main_div:
        return {"time": dt_gmt7, "content": "❌ Content section not found."}

    text_div = main_div.find('div', class_='text-formatted')
    if not text_div:
        return {"time": dt_gmt7, "content": "❌ Text block not found."}
    
    # Get all non-empty <p> tags, skip ones inside ads, embeds
    paragraphs = []
    for p in text_div.find_all('p', recursive=True):
        if p.find_parent(['div', 'iframe'], class_=['ads__container', 'story_ad_container', 'embedcode']):
            continue  # skip ads or embeds
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    
    paragraphs = "\n\n".join(paragraphs)

    return(dt_gmt7, paragraphs)