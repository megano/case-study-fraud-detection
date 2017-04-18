from flask import Flask, request
from sqlalchemy import create_engine
import psycopg2
import pandas as pd
import numpy as np
import datetime
import json
import random
import cPickle as pickle
from predict import PredictFraud
from pandas.io import sql
import urllib2
# pd.set_option('display.max_columns', None)
app = Flask(__name__)


def create_table(cur):
    cur.execute(
        '''CREATE TABLE fraud (
            id serial PRIMARY KEY,
            approx_payout_date integer,
            body_length integer,
            channels integer,
            country text,
            currency text,
            delivery_method double precision,
            description text,
            email_domain text,
            event_created integer,
            event_end integer,
            event_published integer,
            event_start integer,
            fb_published integer,
            gts double precision,
            has_analytics integer,
            has_header double precision,
            has_logo integer,
            listed text,
            name text,
            name_length integer,
            num_order integer,
            num_payouts integer,
            object_id integer,
            org_desc text,
            org_facebook double precision,
            org_name text,
            org_twitter double precision,
            payee_name text,
            payout_type text,
            sale_duration double precision,
            sale_duration2 double precision,
            show_map integer,
            user_age integer,
            user_created integer,
            user_type integer,
            venue_address text,
            venue_country text,
            venue_latitude double precision,
            venue_longitude double precision,
            venue_name text,
            venue_state text,
            ticket_types text,
            previous_payouts text,
            fraud_probability double precision);
        ''')
    conn.commit()
    conn.close()


def insert_db(df, engine, table='fraud'):
    df.to_sql(table, engine, if_exists='append', index=0)


def make_prediction():
    engine = create_engine(
        'postgresql://aymericflaisler@localhost:5432/fraud_prediction')
    # do the prediction
    example_path = './data/test_script_example.json'
    model_path = './data/model.pkl'
    Pred = PredictFraud(
        model_path, 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point', is_json=1)
    X_prep = Pred.fit()
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict_proba(X_prep)[0, 1]
    df = Pred.read_entry()
    df['fraud_probability'] = y_pred
    insert_db(df, engine, table='fraud')
    return X_prep, y_pred

# Flask can respond differently to various HTTP methods
# By default only GET allowed, but you can change that using the methods
# argument


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        # json_dict = request.get_json()
        # description = json_dict['description']
        # json.load(open(example_path))['description']
        # data = {'description': description}
        return "nothing here..."
    else:
        # response = urllib2.urlopen(
        #     'http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
        # raw_json = json.load(response)
        X, y = make_prediction()
        return str(X) + " prediction: " + str(y)


if __name__ == "__main__":
    app.run(debug=True)
