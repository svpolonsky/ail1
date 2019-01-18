# python report.py
# 2019-01-09: I will work on export.py instead. All data massage will be done in Excel for a while
import os
import pandas as pd
from crmdb import *

conn=connect_CRMDB()
df=pd.read_sql_query("SELECT * FROM transactions",conn)
conn.close()
#print(df.columns.tolist())
# convert string dates to datetime
df['transaction_datetime']=pd.to_datetime(df['transaction_datetime'])
#mask=(df['transaction_datetime']>'2018-12-04')
#print(df.loc[mask])
grouped=df.groupby(pd.Grouper(key='transaction_datetime',freq='W'))
for name,group in grouped:
    print(name)
    #print(group)
