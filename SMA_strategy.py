# SMA backtesting class

import numpy as np
import pandas as pd
from scipy.optimize import brute
from matplotlib.pylab import mpl, plt


class SMAVectorBacktester(object):
    """Class for the vectorized backtesting of SMA-based trading strategies.
    Attributes
    ==========
    Feature: str
        'Close' feature with which to work
    SMA1: int
        time window in days for shorter SMA
    SMA2: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new SMA parameters
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates SMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimization for the two SMA parameters
    """

    def __init__(self, Feature, SMA1, SMA2, start, end):
        self.Feature = Feature
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        """ Retrieves and prepares the data.
        """
        raw = pd.read_csv('strategies_data.csv', index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.Feature])
        raw = raw.loc[self.start:self.end]
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['Close'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['Close'].rolling(self.SMA2).mean()
        self.data = raw

    def set_parameters(self, SMA1=None, SMA2=None):
        """ Updates SMA parameters and resp. time series.
        """
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['Close'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['Close'].rolling(self.SMA2).mean()

    def run_strategy(self):
        """ Backtests the trading strategy.
        """
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        """ Plots the cumulative performance of the trading strategy compared to the symbol.
        """
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        plt.style.use('seaborn')
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['font.family'] = 'serif'
        title = '%s %s | SMA1=%d, SMA2=%d' % ('GLD', self.Feature, self.SMA1, self.SMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))
        plt.savefig(title+'.png')
        plt.show()

    def update_and_run(self, SMA):
        """ Updates SMA parameters and returns negative absolute performance
        (for minimization algorithm).
        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        """
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        """ Finds global maximum given the SMA parameter ranges.
            Parameters
            ==========
            SMA1_range, SMA2_range: tuple
            tuples of the form (start, end, step size)
        """
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)


if __name__ == '__main__':
    smabt = SMAVectorBacktester('Close', 42, 252,'2015-1-2', '2022-05-31')
    print(smabt.run_strategy())
    smabt.set_parameters(SMA1=20, SMA2=100)
    print(smabt.run_strategy())
    print(smabt.optimize_parameters((30, 56, 4), (200, 300, 4)))
    smabt.plot_results()
