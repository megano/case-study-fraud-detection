import json
import numpy as np
import pandas as pd

# Object takes JSON file and prepares it for ML modeling
class DataWrangler(object):
    def get_data(self, filename):
        '''
        In: name of JSON file with nested dictionaries
        Out: data frame
        '''
        with open(filename) as data_file:
            data = json.load(data_file)
        df = pd.DataFrame.from_dict(data)
        return df

    def wrangle_data(self, filename):
        '''
        In: df
        Out: df with new columns including fraud and dummy features from categorical data, NA fields filled.
        '''
        df = self.get_data(filename)
        # Add a 'Fraud' column. If event type is 'fraudster_event', 'fraudster', 'fraudster_att'
        # in acct_type field: True (1), else False (0).
        df['fraud'] = df.acct_type.map(lambda x: 1 if ((x == 'fraudster') | (x == 'fraudster_att') | (x == 'fraudster_event')) else 0)
        # split out different currencies
        df['eur'] = df.currency.map(lambda x: 1 if (x == 'eur') else 0)
        df['usd'] = df.currency.map(lambda x: 1 if (x == 'usd') else 0)
        df['gbp'] = df.currency.map(lambda x: 1 if (x == 'gbp') else 0)
        # payment types
        df['ach'] = df.payout_type.map(lambda x: 1 if x == 'ACH' else 0)
        df['check'] = df.payout_type.map(lambda x: 1 if x == 'CHECK' else 0)
        df['blank_payment'] = np.where(df['payout_type']=="", 1, 0)
        # was host paid out for prior events
        df['has_past_payouts'] = df.num_payouts.map(lambda x: 1 if x > 0 else 0)
        # is venue name field filled out - 0 or 1
        df['has_venue_name'] = np.where(pd.isnull(df['venue_name']), 0, 1)
        # is org name field filled out - 0 or 1c
        df['has_org_name'] = np.where(pd.isnull(df['org_name']), 0, 1)
        # is there a number of orders listed - 0 or 1
        df['has_num_order'] = np.where(df['num_order']>0, 1, 0)
        # is logo name field filled out - 0 or 1
        df['has_logo'] = np.where(pd.isnull(df['has_logo']), 0, 1)
        # has value for number of twitter - 0 or 1
        df['has_twitter'] = np.where(pd.isnull(df['org_twitter']), 0, 1)
        # has value for number of twitter - 0 or 1
        df['has_fb'] = np.where(pd.isnull(df['org_facebook']), 0, 1)
        self.df = df.fillna(0)
        df.reset_index(drop=1, inplace=1)
        return df
