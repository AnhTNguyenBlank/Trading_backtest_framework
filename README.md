# Intraday_trading
 
This is a general frame work for backtesting trading strategy including:

* Support functions for plotting charts, finding and visualizing supports/ resistances, finding and visualizing extremum,

* An example notebook of how to use the classes and functions,

* A base class for strategy, backtest to use in other projects.

Some different features that this framework has:

* In this backtesting repo, it replicates real-life trading as in we tend to increase our lot even though we only backtested with 0.01. To incorporate that, I have set up a mechanism, basically the lot will increase with the increment of 0.01 (can be modified) for every time that we double our capital (can also be modified),

* In addition, there will also be a max_existing_positions parameter to ensure our margins, since trading forex, coins and gold involve a very high margins (1:100 or even 1:2000). This mechanism will be updated with the calculation of margin, free margin to better mimic real profit,

* The outputs of the backtest class including dataframes with positions (df_result) and balances (df_balance), which can be used for in-sample optimization. 