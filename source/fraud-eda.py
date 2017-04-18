import json
import numpy as np
import pandas as pd

with open('data.json') as data_file:
  data = json.load(data_file)

df = pd.DataFrame.from_dict(data)

'''
[10:08]
df['fraud'] = df.acct_type.map(lambda x: 1 if ((x == 'fraudster') | (x == 'fraudster_att') | (x == 'fraudster_event')) else 0)

gts = Gross sales are the grand total of all sale transactions reported in a period
'''

# # convert time values to time type
# df['approx_payout_date'] = df['approx_payout_date'].map(lambda x: datetime.datetime.fromtimestamp(x))
# del df['ticket_types']
