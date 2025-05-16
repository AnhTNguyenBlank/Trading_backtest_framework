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

from datetime import datetime
import pytz
import sys


#=======================ALPHAS=======================


class Base_Alpha:
    def __init__(self, strat, prepare_data):
        # Prepare function
        self.prepare_data = prepare_data
        # Strategy function
        self.strat = strat 

        self.config = None

    def signal(self, df_observe):
        '''
        Return the tested dataframe with additional columns: signal (buy/ sell), stop loss, take profit.
        In case eval = False --> live trading, return only the vector of (signal, stop_loss, take_profit).
        '''
        df_prepare = self.prepare_data(df_observe)           
        # Apply the strategy to the dataframe and return the result in IS
        df_result = df_prepare.copy().apply(self.strat, axis=1)

        return(df_result)



class RSI_strat(Base_Alpha):
    def __init__(self, config):
        super().__init__(
            strat = self._RSI_strat,
            prepare_data = self._prepare_data
        )  # Pass the strategy function to the parent constructor
        
        '''
        config = {'RSI_param': 14, 'base_SL': 10, 'base_TP': 20}
        '''
        self.RSI_param = config['RSI_param']
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        
        
    def _prepare_data(self, df_observe):
        df_observe['RSI'] = ta.momentum.RSIIndicator(df_observe['CLOSE'], window = self.RSI_param).rsi()
        return(df_observe)

    def _RSI_strat(self, row):
        '''Strategy for RSI, calculates the signal based on the RSI value.'''
        if row['RSI'] < 30:
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif row['RSI'] > 70:
            return pd.Series([-1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Sell signal
        else:
            return pd.Series([0, 0, 0], index=['SIGNAL', 'SL', 'TP']) # No action (neutral)
        

