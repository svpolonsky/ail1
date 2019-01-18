# python export.py
# Anaconda doesn't have openpyxl (works with Excel), use "pip install openpyxl" to add this module

import os
import pandas as pd
from crmdb import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, help="input database file name")
ap.add_argument("-o", "--output", type=str, help="output excel file name")
args = vars(ap.parse_args())

if args["input"] is not None:
    input=args["input"]
else:
    input="crm.db"

if args["output"] is not None:
    output=args["output"]
else:
    output="crm.xlsx"


conn=connect_CRMDB(dbname=input)
df=pd.read_sql_query("SELECT * FROM transactions",conn)
conn.close()
# convert string dates to datetime
df['transaction_datetime']=pd.to_datetime(df['transaction_datetime'])
# add "helper" column for constructing Pivot Table in Excel
df['helper']=1
path=os.path.join(get_CRMDB_dir(),output)
print(path)
writer=pd.ExcelWriter(path)
df.to_excel(writer,'Sheet1',index=False)
writer.save()
