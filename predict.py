import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
from RandomForest_fraud_detection import RFmodel
import urllib2
pd.set_option('display.max_columns', None)


class PredictFraud(object):
    '''
    Reads in a single example from test_script_examples, unpickles the model, predicts the
    label, and outputs the label probability
    '''

    def __init__(self, model_path, example_path, is_json=0):
        self.model_path = model_path
        self.example_path = example_path
        self.is_json = is_json

    def read_entry(self):
        '''
        Read single entry from http://galvanize-case-study-on-fraud.herokuapp.com/data_point
        '''
        if self.is_json != 0:
            response = urllib2.urlopen(self.example_path)
            d = json.load(response)
        else:
            with open(self.example_path) as data_file:
                d = json.load(data_file)
        df = pd.DataFrame()
        df_ = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems() if (
            k != 'ticket_types') and (k != 'previous_payouts')]))
        df_['ticket_types'] = str(d['ticket_types'])
        df_['previous_payouts'] = str(d['previous_payouts'])
        df = df.append(df_)
        df.reset_index(drop=1, inplace=1)
        df.fillna(0, inplace=1)
        self.example = df
        return df

    def fit(self):
        '''
        Load model with cPickle
        '''
        self.read_entry()

        X_prep = RFmodel().prepare_data(self.example)
        # print model.predict(X_prep)
        return X_prep

    def predict(self):
        return self.model.predict_proba(self.X_prep)


if __name__ == '__main__':
    example_path = './data/test_script_example.json'

    model_path = './data/model.pkl'
    # test = mdPred.read_entry()
    # RFmodel().prepare_data(test)
    mdPred = PredictFraud(
        model_path, example_path)
    df = mdPred.read_entry()
    X_prep = mdPred.fit()
    np.shape(X_prep)
    model = pickle.load(open(model_path, 'rb'))
    model.predict_proba(X_prep)
    y_pred = model.predict(X_prep)
