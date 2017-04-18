import pandas as pd
import numpy as np
import json

def clv_calc(df, age, profit, d, r):
    """calculate and return median and average clv

    Parameters
    ----------
    data_frame : data frame
    age : age column
    profit : profit column
    d : discount rate
    r : retention rate

    Returns
    -------
    avg_clv    : scalar value for clv
    """
    #set user age >0
    user_age = df[age].replace(0, 1)
    GC = df[profit]/(user_age/365)
    #estimate the retention rate as 0.5
    #yearly discount rate = 0.08
    clv = GC * (1/(1+d-r))
    return clv.median(), clv.mean()


if __name__=="__main__":

    json_dic = open('/Users/etownbetty/Documents/Galvanize/Fraud/files/data.json').read()
    data = json.loads(json_dic)
    df = pd.DataFrame.from_dict(data)

    #calculate the median of gross ticket sales
    med_gts = df['gts'].median()
    #subtract for price of phone call to prospective fraudster
    phone_call_price = 10
    benefit = med_gts-phone_call_price
    #cost-benefit matrix for single event fraud
    cb_single_event_savings_cost = np.array([[benefit, -phone_call_price], [0, 0]])
    med_clv, mean_clv = avg_clv(df, 'user_age', 'gts', 0.08, 0.5)
    #use the median clv and calculate the fraction of profit (0.055) for the fraction we lose for
        # incorrect classification
    profit_fraction = 0.055
    event_lost_revenue = med_gts*profit_fraction
    med_clv_lost_revenue = med_clv*profit_fraction
    cb_clv_cost_revenue_cost = np.array([[0, -event_lost_revenue], [-med_clv_lost_revenue, 0]])
