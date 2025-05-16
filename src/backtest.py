import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)

import ta
import matplotlib.pyplot as plt

plt.style.use('classic')
from datetime import datetime, timedelta



class Backtest_report:
    def __init__(self, alpha, 
                 df_is,
                 base_SL = 10, base_TP = 20, 
                 max_existing_positions = 3, init_vol = 0.01, incre_vol = 0.01, max_vol = 0.1,
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
        df_signal_is = self.alpha.signal(self.df_is)
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
