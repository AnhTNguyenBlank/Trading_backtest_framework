import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)

import ta
import matplotlib.pyplot as plt

plt.style.use('classic')
from datetime import datetime, timedelta


class Base_Asset:
    def __init__(self, asset = 'XAUUSD'):
        self.asset = asset
        
    def _cal_margins(self):
        '''
        margins_level = 1/100 #1/200 #1/1000 #1/2000
        '''
        if self.asset == 'XAUUSD':
            # Base stats
            self.lot_per_asset = 0.01
            self.profit_multiplier = 100
        elif self.asset == 'BTCUSD':
            self.lot_per_asset = 10
            self.profit_multiplier = 1
        elif self.asset == 'EURUSD':
            self.lot_per_asset = 1/200000
            self.profit_multiplier = 100000
        



class Backtest_report:
    def __init__(self, 
                 alpha, df_signal, apply_alpha, 
                 adjust_SL, beta, df_sl_beta, apply_beta,
                 df_is,
                 base_SL = 10, base_TP = 20, 
                 max_existing_positions = 3, init_vol = 0.01, incre_vol = 0.01, max_vol = 0.1,
                 init_cap = 1000, incre_cap = 2, asset = 'XAUUSD', margins_level = 1/100, 
                 re_allocation = True
                 ):
        
        self.alpha = alpha # alpha class
        self.apply_alpha = apply_alpha # bool
        self.df_signal = df_signal # df with 3 columns ['SIGNAL', 'SL', 'TP']

        self.adjust_SL = adjust_SL # bool
        self.beta = beta # beta class
        self.apply_beta = apply_beta # bool
        self.df_sl_beta = df_sl_beta # df with 2 columns ['TIME', 'SL_beta']

        self.df_is = df_is # data
        self.init_vol = init_vol # initial volume
        self.re_allocation = re_allocation
        self.incre_vol = incre_vol # increment of volume
        self.max_vol = max_vol # max value of volume
        self.init_cap = init_cap # init capital
        self.incre_cap = incre_cap # the condition of capital if increasing the trading volume
        
        self.asset = Base_Asset(asset = asset)
        self.asset._cal_margins()
        self.margins_level = margins_level

        self.max_existing_positions = max_existing_positions
        self.base_SL = base_SL
        self.base_TP = base_TP    

        self.df_position_tracking = None
        self.df_balance_tracking = None
        self.df_balance_tracking_new = None

        self.df_position_tracking_beta = None
        self.df_balance_tracking_beta = None
        self.df_balance_tracking_beta_new = None
        

    def _track_positions(self, df_is, alpha):

        if self.apply_alpha:
            df_signal = alpha.signal(df_is)
        else:
            df_signal = self.df_signal

        df_position_tracking = df_is[['CLOSE']].copy().merge(df_signal, how = 'left', left_index = True, right_index = True)

        df_position_tracking['VOL'] = self.init_vol
        signals = df_position_tracking[df_position_tracking['SIGNAL'] != 0]

        for ids in range(len(signals)):
            s = signals.iloc[ids, :]
            df_temp = df_position_tracking.loc[
                (df_position_tracking.index > s.name) & 
                (df_position_tracking.index <= s.name + pd.Timedelta(hours = 4)) & 
                (
                    ((df_position_tracking['CLOSE'] - s.CLOSE)*s.SIGNAL*(s.VOL*self.asset.profit_multiplier) < s.SL) | 
                    ((df_position_tracking['CLOSE'] - s.CLOSE)*s.SIGNAL*(s.VOL*self.asset.profit_multiplier) > s.TP) |
                    (df_position_tracking.index == s.name + pd.Timedelta(hours = 4))  # Force close at 4 hours
                ),
                :
            ]

            if len(df_temp) != 0:
                df_position_tracking.loc[s.name, 'TIME_CLOSE_POSITION'] = df_temp.index[0]
            else:
                df_position_tracking.loc[s.name, 'TIME_CLOSE_POSITION'] = pd.NaT

        # Create the column to determine valid position (total number of positions, margins)
        df_position_tracking['FLAG_VALID_POSITION'] = np.where((df_position_tracking['SIGNAL'] != 0) & (df_position_tracking['TIME_CLOSE_POSITION'].isnull() == False), 1, 0)
        df_position_tracking['USED_MARGINS'] = -1 * df_position_tracking['VOL'] * np.abs(df_position_tracking['SIGNAL']) * self.margins_level * df_position_tracking['CLOSE'] / self.asset.lot_per_asset

        df_position_tracking['TIME_CLOSE_POSITION'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['TIME_CLOSE_POSITION'], pd.NaT)
        df_position_tracking['TIME_CLOSE_POSITION'] = pd.to_datetime(df_position_tracking['TIME_CLOSE_POSITION'])

        # Finalize the ORIGINAL position tracking
        df_position_tracking = df_position_tracking.merge(df_position_tracking[['CLOSE']], 
                                                how = 'left', 
                                                left_on = 'TIME_CLOSE_POSITION', 
                                                right_index = True,
                                                suffixes = ('_open', '_close')
                                                )
        df_position_tracking['TIME_OPEN_POSITION'] = df_position_tracking.index.copy()

        # Calculate PNL
        df_position_tracking['PNL'] = (df_position_tracking['CLOSE_close'] - df_position_tracking['CLOSE_open'])*df_position_tracking['SIGNAL']*df_position_tracking['FLAG_VALID_POSITION']*(df_position_tracking['VOL']*self.asset.profit_multiplier)
        df_position_tracking['PNL'] = np.where(df_position_tracking['PNL'] < df_position_tracking['SL'], df_position_tracking['SL'], 
                                    np.where(df_position_tracking['PNL'] > df_position_tracking['TP'], df_position_tracking['TP'], 
                                            np.where(df_position_tracking['PNL'].isnull() == True, 0, df_position_tracking['PNL'])))
        df_position_tracking['PNL'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['PNL'], 0)


        # Adjust the FLAG_VALID_POSITION according to max_existing_positions
        signals = df_position_tracking[(df_position_tracking['SIGNAL'] != 0) & (df_position_tracking['TIME_CLOSE_POSITION'].isnull() == False)]
        existing_positions = signals[:self.max_existing_positions].copy()

        for ids in range(len(signals)):
            if ids >= self.max_existing_positions:

                s = signals.iloc[ids, :]

                # Condition: reject if signal happens before earliest close
                if (s.name < existing_positions['TIME_CLOSE_POSITION'].min()) & (len(existing_positions) == self.max_existing_positions):
                    df_position_tracking.loc[s.name, 'FLAG_VALID_POSITION'] = 0
                else:
                    df_position_tracking.loc[s.name, 'FLAG_VALID_POSITION'] = 1

                    # Remove already closed positions
                    existing_positions = existing_positions.loc[
                        existing_positions['TIME_CLOSE_POSITION'] > s.name, :
                    ]

                    row = pd.DataFrame([s], index=[s.name])
                    if not row.isna().all(axis=None):
                        existing_positions = pd.concat([existing_positions, row], axis=0)
                        
                    existing_positions['TIME_CLOSE_POSITION'] = pd.to_datetime(existing_positions['TIME_CLOSE_POSITION'])

        df_position_tracking['PNL'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['PNL'], 0)
        df_position_tracking['POSITION_ID'] = df_position_tracking['TIME_OPEN_POSITION'].astype('int64')//10**9


        # Apply beta (if any)
        if self.adjust_SL:
            if self.apply_beta:
                df_position_tracking_beta = self.beta.adjust_SL(df_position = df_position_tracking.copy())
            else:
                df_position_tracking_beta = self.df_sl_beta

            signals_beta = df_position_tracking_beta[df_position_tracking_beta['SIGNAL'] != 0]

            # Determine the time hitting TP/ SL - time close position with beta
            for ids in range(len(signals_beta)):
                s = signals_beta.iloc[ids, :]
                if not pd.isna(s.SL_beta):
                    df_temp = df_position_tracking_beta.loc[
                        (df_position_tracking_beta.index > s.TIME_ADJUST_SL) & 
                        (df_position_tracking_beta.index <= s.name + pd.Timedelta(hours = 4)) & 
                        (
                            ((df_position_tracking_beta['CLOSE_open'] - s.CLOSE_open)*s.SIGNAL*(s.VOL*self.asset.profit_multiplier) < s.SL_beta) | 
                            ((df_position_tracking_beta['CLOSE_open'] - s.CLOSE_open)*s.SIGNAL*(s.VOL*self.asset.profit_multiplier) > s.TP) |
                            (df_position_tracking_beta.index == s.name + pd.Timedelta(hours = 4))  # Force close at 4 hours
                        ),
                        :
                    ]

                    if len(df_temp) != 0:
                        df_position_tracking_beta.loc[s.name, 'TIME_CLOSE_POSITION_beta'] = df_temp.index[0]
                    else:
                        df_position_tracking_beta.loc[s.name, 'TIME_CLOSE_POSITION_beta'] = pd.NaT

            df_position_tracking_beta['TIME_CLOSE_POSITION_beta'] = pd.to_datetime(np.where(df_position_tracking_beta['TIME_ADJUST_SL'].isnull(), df_position_tracking_beta['TIME_CLOSE_POSITION'], df_position_tracking_beta['TIME_CLOSE_POSITION_beta']))

            signals_beta = df_position_tracking_beta[df_position_tracking_beta['SIGNAL'] != 0]
            df_position_tracking_beta['SL_beta'] = np.where(df_position_tracking_beta['SL_beta'].isnull(), 0, df_position_tracking_beta['SL_beta'])

            # Finalize the BETA position tracking
            df_position_tracking_beta = df_position_tracking_beta.merge(df_position_tracking_beta[['CLOSE_open']], 
                                                    how = 'left', 
                                                    left_on = 'TIME_CLOSE_POSITION_beta', 
                                                    right_index = True,
                                                    suffixes = ('', '_beta')
                                                    )

            df_position_tracking_beta.columns = ['CLOSE_open', 'SIGNAL', 'SL', 'TP', 'VOL',	'TIME_CLOSE_POSITION', 'FLAG_VALID_POSITION', 'USED_MARGINS', 'CLOSE_close', 'TIME_OPEN_POSITION', 'PNL', 'POSITION_ID', 'TIME_ADJUST_SL', 'SL_beta', 'TIME_CLOSE_POSITION_beta', 'CLOSE_close_beta']

            # Calculate PNL for beta
            df_position_tracking_beta['PNL_beta'] = (df_position_tracking_beta['CLOSE_close_beta'] - df_position_tracking_beta['CLOSE_open'])*df_position_tracking_beta['SIGNAL']*df_position_tracking_beta['FLAG_VALID_POSITION']*(df_position_tracking_beta['VOL']*self.asset.profit_multiplier)
            df_position_tracking_beta['PNL_beta'] = np.where(df_position_tracking_beta['PNL_beta'].isnull() == True, 0, 
                                                            np.where(df_position_tracking_beta['PNL_beta'] > df_position_tracking_beta['TP'], df_position_tracking_beta['TP'], 
                                                                    np.where(df_position_tracking_beta['PNL_beta'] < df_position_tracking_beta['SL_beta'], df_position_tracking_beta['SL_beta'], df_position_tracking_beta['PNL_beta'])))

            df_position_tracking_beta['PNL_beta'] = np.where(df_position_tracking_beta['TIME_ADJUST_SL'].isnull(), df_position_tracking_beta['PNL'], df_position_tracking_beta['PNL_beta'])
            df_position_tracking_beta['PNL_beta'] = np.where(df_position_tracking_beta['FLAG_VALID_POSITION'] == 1, df_position_tracking_beta['PNL_beta'], 0)


            # Adjust the FLAG_VALID_POSITION according to max_existing_positions for BETA
            existing_positions = signals_beta[:self.max_existing_positions].copy()
            df_position_tracking_beta['FLAG_VALID_POSITION_beta'] = df_position_tracking_beta['FLAG_VALID_POSITION'].copy()

            for ids in range(len(signals)):
                if ids >= self.max_existing_positions:

                    s = signals.iloc[ids, :]

                    # Condition: reject if signal happens before earliest close
                    if (s.name < existing_positions['TIME_CLOSE_POSITION'].min()) & (len(existing_positions) == self.max_existing_positions):
                        df_position_tracking_beta.loc[s.name, 'FLAG_VALID_POSITION'] = 0
                    else:
                        df_position_tracking_beta.loc[s.name, 'FLAG_VALID_POSITION'] = 1

                        # Remove already closed positions
                        existing_positions = existing_positions.loc[
                            existing_positions['TIME_CLOSE_POSITION'] > s.name, :
                        ]

                        row = pd.DataFrame([s], index=[s.name])
                        if not row.isna().all(axis=None):
                            existing_positions = pd.concat([existing_positions, row], axis=0)
                            
                        existing_positions['TIME_CLOSE_POSITION'] = pd.to_datetime(existing_positions['TIME_CLOSE_POSITION'])


            df_position_tracking_beta['PNL_beta'] = np.where(df_position_tracking_beta['FLAG_VALID_POSITION_beta'] == 1, df_position_tracking_beta['PNL_beta'], 0)

        if self.adjust_SL:
            return(df_position_tracking_beta)
        else:
            return(df_position_tracking)

    def _cal_balance(self, df_is, df_position_tracking, 
                     cols_names = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']
                     ):
        # Calculate balance
        df_tracking = pd.DataFrame(index = df_is.index)

        df_tracking = df_tracking.merge(
            pd.pivot_table(
                df_position_tracking,
                index = cols_names[0],
                values = [cols_names[1], cols_names[2], cols_names[3]],
                aggfunc = 'sum'
            ),
            how = 'left',
            left_index = True,
            right_index = True
        )

        df_tracking.columns = ['OPEN_POSITION_ID', 'USED_MARGINS', 'OPEN_VOL']

        df_tracking = df_tracking.merge(
            pd.pivot_table(
                df_position_tracking,
                index = cols_names[4],
                values = [cols_names[1], cols_names[2], cols_names[5]],
                aggfunc = 'sum'
            ),
            how = 'left',
            left_index = True,
            right_index = True
        )

        df_tracking.columns = ['OPEN_POSITION_ID', 'USED_MARGINS', 'OPEN_VOL', 'PNL', 'ADDITIONAL_MARGINS', 'CLOSED_VOL']
        df_tracking = df_tracking.fillna(0).infer_objects(copy=False)

        df_tracking['OPEN_POSITION_ID'] = df_tracking['OPEN_POSITION_ID'].astype(str)
        df_tracking['CLOSED_VOL'] = df_tracking['CLOSED_VOL']*-1
        df_tracking['ADDITIONAL_MARGINS'] = df_tracking['ADDITIONAL_MARGINS']*-1

        df_tracking['NUM_CURRENT_POSITIONS'] = df_tracking['OPEN_VOL'].cumsum() + df_tracking['CLOSED_VOL'].cumsum()
        df_tracking['NUM_CURRENT_POSITIONS'] = df_tracking['NUM_CURRENT_POSITIONS'].astype(float)

        df_tracking['BALANCE'] = df_tracking['PNL'].cumsum() + self.init_cap
        df_tracking['FREE_MARGINS'] = df_tracking['BALANCE'] + df_tracking['USED_MARGINS'].cumsum() + df_tracking['ADDITIONAL_MARGINS'].cumsum()
        df_tracking['DRAWDOWN'] = np.where(df_tracking['PNL'] < 0, df_tracking['PNL']/(df_tracking['BALANCE'] - df_tracking['PNL']), 0)
        

        return(df_tracking)

    def _re_balance(self, df_position_tracking, df_balance_tracking):
        # Reallocating the capital
        
        idt = self.df_balance_tracking.index[0]
        cap = self.init_cap
        vol = self.init_vol

        df_position_tracking['NEW_VOL'] = df_position_tracking['VOL']
        df_position_tracking['NEW_USED_MARGINS'] = df_position_tracking['USED_MARGINS']
        df_position_tracking['NEW_PNL'] = df_position_tracking['PNL']

        for idd, d in enumerate(df_balance_tracking.index): 
            if df_balance_tracking.loc[d, 'BALANCE']/cap >= self.incre_cap:
                idt =  d
                cap *= self.incre_cap
                vol += self.incre_vol

                if vol >= self.max_vol:
                    vol = self.max_vol

                df_position_tracking.loc[df_position_tracking['TIME_OPEN_POSITION'] > idt, 'NEW_VOL'] = vol
                df_position_tracking['NEW_USED_MARGINS'] = -1 * df_position_tracking['NEW_VOL'] * np.abs(df_position_tracking['SIGNAL']) * self.margins_level * df_position_tracking['CLOSE_open'] / self.asset.lot_per_asset
                
                df_position_tracking['NEW_PNL'] = (df_position_tracking['CLOSE_close'] - df_position_tracking['CLOSE_open'])*df_position_tracking['SIGNAL']*df_position_tracking['FLAG_VALID_POSITION']*(df_position_tracking['NEW_VOL']*self.asset.profit_multiplier)
                df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['NEW_PNL'] < df_position_tracking['SL']*(df_position_tracking['NEW_VOL']/df_position_tracking['VOL']), df_position_tracking['SL']*(df_position_tracking['NEW_VOL']/df_position_tracking['VOL']), 
                                            np.where(df_position_tracking['NEW_PNL'] > df_position_tracking['TP']*(df_position_tracking['NEW_VOL']/df_position_tracking['VOL']), df_position_tracking['TP']*(df_position_tracking['NEW_VOL']/df_position_tracking['VOL']), 
                                                    np.where(df_position_tracking['NEW_PNL'].isnull() == True, 0, df_position_tracking['NEW_PNL'])))
                df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['NEW_PNL'], 0)
        

        df_position_tracking['NEW_VOL'] = np.where(df_position_tracking['NEW_VOL'].isnull(), df_position_tracking['VOL'], df_position_tracking['NEW_VOL'])
        df_position_tracking['NEW_USED_MARGINS'] = np.where(df_position_tracking['NEW_USED_MARGINS'].isnull(), df_position_tracking['USED_MARGINS'], df_position_tracking['NEW_USED_MARGINS'])
        df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['NEW_PNL'] == 0, df_position_tracking['PNL'], df_position_tracking['NEW_PNL'])


        df_temp = df_position_tracking[['TIME_OPEN_POSITION', 'NEW_VOL', 'NEW_USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'NEW_PNL']].copy()
        df_temp.columns = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']    

        df_balance_tracking_new = self._cal_balance(df_is = self.df_is, df_position_tracking = df_temp)
        return(df_position_tracking, df_balance_tracking_new)

    def prepare_report(self):

        df_position_tracking = self._track_positions(
            df_is = self.df_is,
            alpha = self.alpha
        )

        df_balance_tracking = self._cal_balance(
            df_is = self.df_is, 
            df_position_tracking = df_position_tracking.loc[df_position_tracking['FLAG_VALID_POSITION'] == 1, 
                                                            ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']].copy(),
            cols_names = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']
        )

        self.df_position_tracking = df_position_tracking
        self.df_balance_tracking = df_balance_tracking
        
        if self.re_allocation:
            self.df_position_tracking, self.df_balance_tracking_new = self._re_balance(df_position_tracking = df_position_tracking[df_position_tracking['FLAG_VALID_POSITION'] == 1].copy(), 
                                                                                       df_balance_tracking = df_balance_tracking.copy())    

        if self.adjust_SL:
            df_position_tracking_beta = self.df_position_tracking.copy()

            df_position_tracking_beta = df_position_tracking_beta[['TIME_OPEN_POSITION', 'CLOSE_open', 'SIGNAL', 'SL_beta', 'TP', 'VOL', 'TIME_CLOSE_POSITION_beta', 'FLAG_VALID_POSITION_beta', 'USED_MARGINS', 'CLOSE_close', 'TIME_ADJUST_SL', 'PNL_beta', 'POSITION_ID']]
            df_position_tracking_beta.columns = ['TIME_OPEN_POSITION', 'CLOSE_open', 'SIGNAL', 'SL', 'TP', 'VOL', 'TIME_CLOSE_POSITION', 'FLAG_VALID_POSITION', 'USED_MARGINS', 'CLOSE_close', 'TIME_ADJUST_SL', 'PNL', 'POSITION_ID']

            self.df_position_tracking_beta = df_position_tracking_beta.copy()
            self.df_balance_tracking_beta = self._cal_balance(
                df_is = self.df_is, 
                df_position_tracking = df_position_tracking_beta[df_position_tracking_beta['FLAG_VALID_POSITION'] == 1].copy(),
                # cols_names = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION_beta', 'PNL_beta']
            )

            if self.re_allocation:
                self.df_position_tracking_beta, self.df_balance_tracking_beta_new = self._re_balance(df_position_tracking = df_position_tracking_beta[df_position_tracking_beta['FLAG_VALID_POSITION'] == 1].copy(), 
                                                                                                     df_balance_tracking = self.df_balance_tracking_beta)    

    def display_report(self):
        if self.df_position_tracking.empty and self.df_balance_tracking.empty:
            self.prepare_report()

        size = (30, 30)

        fig = plt.figure(figsize = size)

        ax = fig.add_subplot(5, 2, 1)
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

        ax_0 = ax.twinx()
        ax_0.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
        ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, :].index[0], 
                    xmax = self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, :].index[-1], 
                    color='r', linestyles = '--', alpha = 0.3)
        
        ax.legend(bbox_to_anchor = (1, 0.3))
        ax_0.legend(bbox_to_anchor = (1, 0.145))
        ax.set_title('Original Balance and Drawdown (No re-allocation)')
        
        expected_index = [-1, 1]
        # ---- WINNING ----
        win_df = pd.pivot_table(
            self.df_position_tracking[
                (self.df_position_tracking['FLAG_VALID_POSITION'] == 1) &
                (self.df_position_tracking['SIGNAL'] != 0) &
                (self.df_position_tracking['PNL'] >= 0)
            ],
            index='SIGNAL',
            values='CLOSE_open',
            aggfunc='count'
        )
        win_df = win_df.reindex(expected_index, fill_value=0)        
        if win_df.shape[1] == 0:
            win_df['WINNING_POSITIONS'] = 0
        else:
            win_df.columns = ['WINNING_POSITIONS']
        

        # ---- LOSING ----
        lose_df = pd.pivot_table(
            self.df_position_tracking[
                (self.df_position_tracking['FLAG_VALID_POSITION'] == 1) &
                (self.df_position_tracking['SIGNAL'] != 0) &
                (self.df_position_tracking['PNL'] < 0)
            ],
            index='SIGNAL',
            values='CLOSE_open',
            aggfunc='count'
        )
        lose_df = lose_df.reindex(expected_index, fill_value=0)        
        if lose_df.shape[1] == 0:
            lose_df['LOSING_POSITIONS'] = 0
        else:
            lose_df.columns = ['LOSING_POSITIONS']
        
        df_summary = pd.concat([win_df, lose_df], axis=1)
        df_summary.loc['ALL'] = df_summary.sum()
        df_summary.index = ['SHORT', 'LONG', 'ALL']


        # ---- Percentages ----
        df_summary['TOTAL_POSITIONS'] = df_summary['WINNING_POSITIONS'] + df_summary['LOSING_POSITIONS']
        safe_total = df_summary['TOTAL_POSITIONS'].replace(0, 1)
        df_summary['PERC_WINNING_POSITIONS'] = df_summary['WINNING_POSITIONS'] / safe_total
        df_summary['PERC_LOSING_POSITIONS']   = df_summary['LOSING_POSITIONS'] / safe_total

        # ---- Bar Plot ----
        ax = fig.add_subplot(5, 2, 3)

        bars1 = ax.bar(df_summary.index, df_summary['PERC_WINNING_POSITIONS'],
                    color='green', label='WINNING_POSITIONS')

        bars2 = ax.bar(df_summary.index, df_summary['PERC_LOSING_POSITIONS'],
                    bottom=df_summary['PERC_WINNING_POSITIONS'],
                    color='red', label='LOSING_POSITIONS')

        win_labels  = df_summary['WINNING_POSITIONS'].astype(int)
        lose_labels = df_summary['LOSING_POSITIONS'].astype(int)
        win_labels  = win_labels.where(win_labels != 0, '')
        lose_labels = lose_labels.where(lose_labels != 0, '')

        ax.bar_label(bars1, labels = win_labels, label_type = 'center')
        ax.bar_label(bars2, labels = lose_labels, label_type = 'center')

        # ax.legend(bbox_to_anchor=(1.3, 1))
        ax.set_title('Distribution of winning/ losing positions')

        ax = fig.add_subplot(5, 2, 5)
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'],
                label='NUM_CURRENT_POSITIONS')
        # ax.legend(bbox_to_anchor=(1.15, 1))
        ax.set_title('Number of positions')


        if self.re_allocation:
            ax = fig.add_subplot(5, 2, 7)

            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

            ax_0 = ax.twinx()
            ax_0.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
            ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, :].index[0], 
                        xmax = self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, :].index[-1], 
                        color='r', linestyles = '--', alpha = 0.3)
            
            ax.legend(bbox_to_anchor = (1.15, 1))
            ax_0.legend(bbox_to_anchor = (1.15, 0.8))    
            ax.set_title('New Balance and Drawdown (with re-allocation)')
          
            ax = fig.add_subplot(5, 2, 9)
            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'], color = 'blue', label = 'NUM_CURRENT_POSITIONS')
            # ax.legend(bbox_to_anchor = (1.15, 1))
            ax.set_ylim((0, 0.1))
            ax.set_title('Volume')

        # Add beta
        if self.adjust_SL:
            ax = fig.add_subplot(5, 2, 2)
            ax.plot(self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
            ax.plot(self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

            ax_0 = ax.twinx()
            ax_0.plot(self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
            ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, :].index[0], 
                        xmax = self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, :].index[-1], 
                        color='r', linestyles = '--', alpha = 0.3)
            
            ax.legend(bbox_to_anchor = (1, 0.3))
            ax_0.legend(bbox_to_anchor = (1, 0.145))
            ax.set_title('BETA_Original Balance and Drawdown (No re-allocation)')
            
            expected_index = [-1, 1]
            # ---- WINNING ----
            win_df = pd.pivot_table(
                self.df_position_tracking_beta[
                    (self.df_position_tracking_beta['FLAG_VALID_POSITION'] == 1) &
                    (self.df_position_tracking_beta['SIGNAL'] != 0) &
                    (self.df_position_tracking_beta['PNL'] >= 0)
                ],
                index='SIGNAL',
                values='CLOSE_open',
                aggfunc='count'
            )
            win_df = win_df.reindex(expected_index, fill_value=0)        
            if win_df.shape[1] == 0:
                win_df['WINNING_POSITIONS'] = 0
            else:
                win_df.columns = ['WINNING_POSITIONS']
            
            # ---- LOSING ----
            lose_df = pd.pivot_table(
                self.df_position_tracking_beta[
                    (self.df_position_tracking_beta['FLAG_VALID_POSITION'] == 1) &
                    (self.df_position_tracking_beta['SIGNAL'] != 0) &
                    (self.df_position_tracking_beta['PNL'] < 0)
                ],
                index='SIGNAL',
                values='CLOSE_open',
                aggfunc='count'
            )
            lose_df = lose_df.reindex(expected_index, fill_value=0)        
            if lose_df.shape[1] == 0:
                lose_df['LOSING_POSITIONS'] = 0
            else:
                lose_df.columns = ['LOSING_POSITIONS']
            
            df_summary = pd.concat([win_df, lose_df], axis=1)
            df_summary.loc['ALL'] = df_summary.sum()
            df_summary.index = ['SHORT', 'LONG', 'ALL']


            # ---- Percentages ----
            df_summary['TOTAL_POSITIONS'] = df_summary['WINNING_POSITIONS'] + df_summary['LOSING_POSITIONS']
            safe_total = df_summary['TOTAL_POSITIONS'].replace(0, 1)
            df_summary['PERC_WINNING_POSITIONS'] = df_summary['WINNING_POSITIONS'] / safe_total
            df_summary['PERC_LOSING_POSITIONS']   = df_summary['LOSING_POSITIONS'] / safe_total
            
            # ---- Bar Plot ----
            ax = fig.add_subplot(5, 2, 4)

            bars1 = ax.bar(df_summary.index, df_summary['PERC_WINNING_POSITIONS'],
                        color='green', label='WINNING_POSITIONS')

            bars2 = ax.bar(df_summary.index, df_summary['PERC_LOSING_POSITIONS'],
                        bottom=df_summary['PERC_WINNING_POSITIONS'],
                        color='red', label='LOSING_POSITIONS')

            # Labels (safe even when zero)
            win_labels  = df_summary['WINNING_POSITIONS'].astype(int)
            lose_labels = df_summary['LOSING_POSITIONS'].astype(int)
            win_labels  = win_labels.where(win_labels != 0, '')
            lose_labels = lose_labels.where(lose_labels != 0, '')

            ax.bar_label(bars1, labels = win_labels, label_type = 'center')
            ax.bar_label(bars2, labels = lose_labels, label_type = 'center')
            
            # ax.legend(bbox_to_anchor=(1.3, 1))
            ax.set_title('BETA_Distribution of winning/ losing positions')


            # ---- Your line plot below stays same ----
            ax = fig.add_subplot(5, 2, 6)
            ax.plot(self.df_balance_tracking_beta.loc[self.df_balance_tracking_beta['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'],
                    label='NUM_CURRENT_POSITIONS')
            # ax.legend(bbox_to_anchor=(1.15, 1))
            ax.set_title('Number of positions')


            if self.re_allocation:
                ax = fig.add_subplot(5, 2, 8)

                ax.plot(self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
                ax.plot(self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

                ax_0 = ax.twinx()
                ax_0.plot(self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
                ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, :].index[0], 
                            xmax = self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, :].index[-1], 
                            color='r', linestyles = '--', alpha = 0.3)
                
                ax.legend(bbox_to_anchor = (1.15, 1))
                ax_0.legend(bbox_to_anchor = (1.15, 0.8))    
                ax.set_title('BETA_New Balance and Drawdown (with re-allocation)')
            
                ax = fig.add_subplot(5, 2, 10)
                ax.plot(self.df_balance_tracking_beta_new.loc[self.df_balance_tracking_beta_new['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'], color = 'blue', label = 'NUM_CURRENT_POSITIONS')
                # ax.legend(bbox_to_anchor = (1.15, 1))
                ax.set_ylim((0, 0.1))
                ax.set_title('BETA_Volume')

        plt.tight_layout()
        plt.show()
