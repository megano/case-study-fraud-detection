import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GDBC
from sklearn.ensemble import AdaBoostClassifier as ADR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, \
accuracy_score, recall_score, f1_score, precision_score, roc_curve

from data_wrangling_helper import DataWrangler

# # convert time values to time type
# df['approx_payout_date'] = df['approx_payout_date'].map(lambda x: datetime.datetime.fromtimestamp(x))
# del df['ticket_types']

def random_forest(df):
    columns_to_keep = ['fraud', 'eur', 'usd', 'gbp', 'ach', 'check', \
    'blank_payment', 'has_past_payouts', 'has_venue_name', 'has_org_name', \
    'has_num_order', 'has_logo', 'user_type', 'delivery_method', 'has_twitter',\
     'has_fb', 'has_analytics', 'gts']
    # find and keep columns we want to use for this model
    df_ = df.loc[:, columns_to_keep]
    # set fraud column as variable to predict
    y = df.fraud
    X = df_.values
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y)
    model = RF(n_estimators = 100, max_depth = 5)
    # fit the model
    model.fit(X_train, y_train)
    # make a prediction
    y_pred = model.predict(X_test)

    # print scores
    print 'Accuracy Score: {}'.format(accuracy_score(y_pred, y_test))
    print 'Recall Score: {}'.format(recall_score(y_pred, y_test))
    print 'Precision Score: {}'.format(precision_score(y_pred, y_test))

    '''
    RF
    Accuracy Score: 0.94170153417
    Recall Score: 0.981132075472
    Precision Score: 0.334405144695
    '''

if __name__ == '__main__':
    filename = 'data.json'
    # create an object of class type DataWrangler
    wrangler = DataWrangler()
    # calls wrangle_data (which calls get_data within that function)
    # and returns cleanded df with new columns
    df = wrangler.wrangle_data(filename)
    random_forest(df)
