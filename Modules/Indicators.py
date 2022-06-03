import statistics as stat
import pandas as pd

def trade(candles, stop_loss, take_profit, short=True, long=True, short_trades=None, long_trades=None, pips_digit=4):
    take_profit *= (10**-pips_digit)
    stop_loss *= (10**-pips_digit)
        
    if long:
        results = {}
        for entry_candle in range(len(candles)):
            
            curr_candle = entry_candle
            tp = candles.loc[entry_candle, 'ask_o'] + take_profit
            sl = candles.loc[entry_candle, 'ask_o'] - stop_loss
            
            while curr_candle in range(len(candles)):
                
                if candles.loc[curr_candle, 'bid_l'] <= sl:
                    results[entry_candle] = False
                    break
                elif candles.loc[curr_candle, 'bid_h'] >= tp:
                    results[entry_candle] = True
                    break
                    
                curr_candle += 1
                
        candles['long_result'] = candles.index.map(results)
        
    if short:
        results = {}
        for entry_candle in range(len(candles)):
            
            curr_candle = entry_candle
            tp = candles.loc[entry_candle, 'bid_o'] - take_profit
            sl = candles.loc[entry_candle, 'bid_o'] + stop_loss
            
            while curr_candle in range(len(candles)):
                
                if candles.loc[curr_candle, 'ask_h'] >= sl:
                    results[entry_candle] = False
                    break
                elif candles.loc[curr_candle, 'ask_l'] <= tp:
                    results[entry_candle] = True
                    break
                    
                curr_candle += 1
                
        candles['short_result'] = candles.index.map(results)
    
    return
    
    
def simple_moving_average(candles, price, source, length):
    if source not in ['open', 'high', 'low', 'close']:
        print('invalid source')
        return None
    if type(length) != int or length < 1:
        print('invalid length')
        return None

    source_dict = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c'
    }
    candles[f'SMA{length}_{source}'] = candles[f'{price}_{source_dict[source]}'].rolling(window=length).mean()

    return


def exponential_moving_average(candles, source, length, price='mid'):
    if source not in ['open', 'high', 'low', 'close']:
        print('invalid source')
        return None
    if type(length) != int or length < 1:
        print('invalid length')
        return None

    source_dict = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c'
    }
    candles[f'EMA_{length}'] = candles[f'{price}_{source_dict[source]}'].ewm(span=length, adjust=False).mean()
    candles.loc[0:length, [f'EMA_{length}']] = None


def average_true_range(candles, length=14):
    if type(length) != int or length < 1:
        print('invalid length')
        return None

    tr = [abs(candles['mid_h'][0] - candles['mid_l'][0])]
    for i in range(1, len(candles.index)):
        a = abs(candles['mid_h'][i] - candles['mid_l'][i])
        b = abs(candles['mid_h'][i] - candles['mid_l'][i - 1])
        c = abs(candles['mid_h'][i - 1] - candles['mid_l'][i])
        tr.append(max(a, b, c))
    atr = [stat.mean(tr[x - length + 1: x + 1]) if x >= length - 1 else None for x in range(len(candles.index))]
    candles[f'atr_{length}'] = atr

    return


def relative_strength_index(candles, length=14, price='mid'):
    change = [(candles[f'{price}_c'][row] - candles[f'{price}_o'][row]) for row in range(len(candles))]
    pos = [max(x, 0) for x in change]
    neg = [abs(min(x, 0)) for x in change]
    cag = stat.mean(pos[0: length])
    cal = stat.mean(neg[0: length])
    rs = [None] * (length - 1)
    rs.append(100 - (100 / (1 + (cag / cal))))

    for x in range((length), len(change)):
        cag = ((cag * 13) + pos[x]) / length
        cal = ((cal * 13) + neg[x]) / length

        rs.append(100 - (100 / (1 + (cag / (cal)))))

    candles[f'rsi_{length}'] = rs


def macd(candles, k=12, d=26, s=9, price='mid'):
    a, b = False, False
    if f'EMA_{k}' not in candles.columns:
        a = True
        exponential_moving_average(candles=candles, source='close', length=k, price='mid')
    if f'EMA_{d}' not in candles.columns:
        b = True
        exponential_moving_average(candles=candles, source='close', length=d, price='mid')

    ma_cd = candles[f'EMA_{k}'] - candles[f'EMA_{d}']
    signal = ma_cd.ewm(span=s, adjust=False).mean()

    candles[f'MACD_{k}_{d}_{s}'] = ma_cd - signal

    if a:
        candles.drop(f'EMA_{k}', 1, inplace=True)
    if b:
        candles.drop(f'EMA_{d}', 1, inplace=True)
    return


def stochastic(candles, k=14, k_smooth=1, d=3, price='mid'):
    k-=1
    candles['stochastic_k'] = float('nan')
    
    for candle in range(k,len(candles)):
        low = min(candles.loc[candle-k : candle, f'{price}_l'])
        high = max(candles.loc[candle-k : candle, f'{price}_h'])
        curr_close = candles.loc[candle, 'mid_c']
        stochastic_k = ((curr_close - low) / (high - low)) * 100
        candles.loc[candle, 'stochastic_k'] = stochastic_k
        
    if k_smooth > 1:
        candles['stochastic_k'] = candles['stochastic_k'].rolling(k_smooth).mean()
        
    candles['stochastic_d'] = candles['stochastic_k'].rolling(d).mean()
        
    return

       
if __name__ == '__main__':
    print('indicator functions')
