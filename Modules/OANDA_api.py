import pandas as pd
import requests
import info
import utils


class oanda_api:
    def __init__(self):
        self.session = requests.Session()
        self.access_token = info.access_token
        self.account_id = info.account_id
        self.static_api = info.static_api
        self.stream_api = info.stream_api
        self.header = {
            'Authorization': f'Bearer {self.access_token}'
        }

    def get_instruments_df(self):
        url = f'{self.static_api}/accounts/{self.account_id}/instruments'
        header = self.header
        code, data = self.session.get(url, params=None, headers=header)
        if code == 200:
            df = pd.DataFrame.from_dict(data['instruments'])
            return df[['name', 'type', 'displayName', 'pipLocation', 'marginRate']]
        else:
            return None

    def fetch_candles(self, pair, count, granularity, prices, to):
        url = f'{self.static_api}/instruments/{pair}/candles'
        #if count >= 5000:
        params = {'count': count, 'granularity': granularity, 'price': prices, 'to': to}
        header = self.header

        response = self.session.get(url, params=params, headers=header)

        if response.status_code == 200:
            return response.json()
        else:
            print(f'Error code: {response.status_code}')
            return None

    # Personal add ons

    def get_candles_df(self, json_response):
        if json_response is None:
            return None
        simple_candles = []
        prices = [x for x in ['mid', 'bid', 'ask'] if x in list(json_response['candles'][0].keys())]
        ohlc = ['o', 'h', 'l', 'c']
        for candle in json_response['candles']:
            if not candle['complete']:
                continue
            new_dict = {'time': candle['time'], 'volume': candle['volume']}
            for p in prices:
                for e in ohlc:
                    new_dict[f'{p}_{e}'] = candle[p][e]
            simple_candles.append(new_dict)
        return pd.DataFrame.from_dict(simple_candles)

    def full_use_candles(self, pair, count, granularity, prices, to=None):
        
        json_candles = self.fetch_candles(pair, count, granularity, prices, to)
        if json_candles is None:
            return None
        candles = self.get_candles_df(json_candles)
        non_cols = ['time', 'volume']
        cols = [x for x in candles.columns if x not in non_cols]
        candles[cols] = candles[cols].apply(pd.to_numeric)
        return candles
    
    def complete_candles_df(self, pair, count, granularity, prices='MBA'):
        if count <= 5000:
            return self.full_use_candles(pair, count, granularity, prices) 
        
        candles = self.full_use_candles(pair, 5000, granularity, prices)
        count -= 5000
        while count > 5000:
            to = candles.loc[0, ['time']]
            older_candles = self.full_use_candles(pair, 5000, granularity, prices, to)
            candles = pd.concat([older_candles, candles], ignore_index=True)
            count -= 5000

        to = candles.loc[0, ['time']]
        older_candles = self.full_use_candles(pair, count, granularity, prices, to)
        candles = pd.concat([older_candles, candles], ignore_index=True)
        return candles


if __name__ == '__main__':
    api = oanda_api()
    candles = api.full_use_candles('EUR_USD', 5, 'H1','M')
    print(candles)
    print(candles.dtypes)
