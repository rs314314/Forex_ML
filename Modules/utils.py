import numpy as np
import pandas as pd


def get_candles_data_filename(pair, granularity):
    return f'candles_data/{pair}_{granularity}.pkl'


def get_instruments_data_filename():
    return 'instruments.pkl'


if __name__ == '__main__':
    print(get_candles_data_filename('EUR_USD', 'H1'))
    print(get_instruments_data_filename())

