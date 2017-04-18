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
pd.set_option('display.max_columns', None)


try:
    conn = psycopg2.connect(
        "dbname='fraud_prediction' user='aymericflaisler' host='localhost' password='1323'")
except:
    print "I am unable to connect to the database"


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


def create_test_table(cur):
    cur.execute(
        '''CREATE TABLE test (
            id serial PRIMARY KEY,
            index
            approx_payout_date integer,
            body_length integer,
            channels integer,
            country text,
            currency text);
        ''')
    conn.commit()
    conn.close()


def test_fill():
                id serial PRIMARY KEY,
                ['approx_payout_date',
                'body_length',
                'channels',
                'country',
                'currency']
    df=pd.DataFrame([[1,2,3,'bla','bla']],columns=['approx_payout_date','body_length','channels','country','currency'])

    # cur.execute('''INSERT INTO test VALUES (%d, %d, %d, %d, %d);'''
    #             % (wordlocation_id, url_id, word_id, location_id))
    # sql.execute('INSERT INTO test VALUES(?, ?, ?,?,?)', engine, params=[('id', 1, 12.2, True)])
    df.to_sql('fraud', engine, if_exists='append', index=0)


if __name__ == '__main__':
    conn = psycopg2.connect(
        "dbname='fraud_prediction' user='aymericflaisler' host='localhost' password='1323'")
    cur = conn.cursor()
    create_table(cur)
    # create_test_table(cur)
    engine = create_engine('postgresql://aymericflaisler@localhost:5432/fraud_prediction')
    df
