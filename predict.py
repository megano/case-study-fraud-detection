import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
pd.set_option('display.max_columns', None)


class PredictFraud(object):
    '''
    Reads in a single example from test_script_examples, unpickles the model, predicts the
    label, and outputs the label probability
    '''

    def read_entry(self, json_path):
        with open(json_path) as data_file:
            d = json.load(data_file)
        df = pd.DataFrame()
        for k, v in d.iteritems():
            df_ = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems() if (
                k != 'ticket_types') and (k != 'previous_payouts')]))
            df_['ticket_types'] = str(d['ticket_types'])
            df_['previous_payouts'] = str(d['previous_payouts'])
            df = df.append(df_)
            df.reset_index(drop=1, inplace=1)
        return df


if __name__ == '__main__':
    json_path = './data/test_script_example.json'
    PredictFraud.read_entry(json_path)
