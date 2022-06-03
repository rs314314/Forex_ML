import pandas as pd
import utils


class Instrument:
    def __init__(self, pair):
        self.name = pair['name']
        self.type = pair['type']
        self.displayName = pair['displayName']
        self.pipLocation = pow(10, pair['pipLocation'])
        self.marginRate = pair['marginRate']

    def __repr__(self):
        return str(vars(self))

    @classmethod
    def get_instruments_df(cls):
        return pd.read_pickle(utils.get_instruments_data_filename())

    @classmethod
    def get_instruments_lists(cls):
        df = cls.get_instruments_df()
        return [Instrument(x) for x in df.to_dict(orient='records')]


if __name__ == '__main__':
    print(Instrument.get_instruments_lists())

