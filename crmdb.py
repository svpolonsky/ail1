# structure of transactions Database
# columns:
# customer_id
# transaction_datetime
# transaction_type (SHOW/DEAL)
#
# How to suggest a deal?
# If N latest transaction are SHOW than suggest a DEAL

# CRM functions

import os
import sqlite3
import datetime

time_format = '%Y-%m-%d %H:%M:%S'

def get_CRMDB_dir():
    return os.path.expanduser("~/CRMDB")


def connect_CRMDB(**kwargs):
    # keyword args: dbname - name of database
    # crmdb: customer relationships management database
    crmdb_name=kwargs.get('dbname','crm.db')
    path=os.path.join(get_CRMDB_dir(),crmdb_name)
    #print("connect crmdb:",path)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    sql = """
    CREATE TABLE transactions (
        id integer PRIMARY KEY,
        customer_id text NOT NULL,
        transaction_datetime text NOT NULL,
        transaction_type text NOT NULL)"""
    try:
        cursor.execute(sql)
    except:
        None
        #print("crmdb: table TRANSACTIONS exists")
    else:
        print("crmdb: created table TRANSACTIONS")
    conn.commit()
    return conn


def record_transaction(customer_id,transaction_type,**kwargs):
    #print("crm:",customer_id,transaction_type)
    conn=connect_CRMDB(**kwargs)
    cursor=conn.cursor()
    sql = """
        INSERT
            INTO transactions (customer_id, transaction_datetime, transaction_type)
            VALUES (?, ?, ?)"""
    system_dt=datetime.datetime.now().strftime(time_format)
    # system datetime override using keyword argument 'dt'
    dt=str(kwargs.get('dt',system_dt))
    cursor.execute(sql, (customer_id, dt,transaction_type))
    conn.commit()
    conn.close()


def last_transaction_time(customer_id):
    # return time of last transcation
    conn=connect_CRMDB()
    cursor=conn.cursor()
    sql="SELECT transaction_datetime FROM transactions WHERE customer_id=? ORDER BY id DESC LIMIT 1"
    cursor.execute(sql,(customer_id,))
    results = cursor.fetchall()
    conn.close()
    if len(results)==1:
        time_str=str(results[0][0])
    else:
        time_str="2000-01-01 00:00:00"
    t=datetime.datetime.strptime(time_str, time_format)
    #print("last_transaction_time:",type(t))
    return t
