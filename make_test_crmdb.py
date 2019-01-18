# I need "test" Customer Relation Database to play with owner reports
import uuid
import numpy as np
import random
random.seed(1)
import datetime
from crmdb import *

# generate customer list
n_customer=10000
customers={}
for i in range(0,n_customer):
    id=str(uuid.uuid1())
    # how often the customer comes?
    customers[id]=random.random()
ids=list(customers.keys())
weights=list(customers.values())
probs=weights/np.sum(weights)

# simulate dayly visits
n_day=365
customer_min=20
customer_max=100
date0=datetime.date(2019,1,1)
for day in range(0,n_day):
    date=date0+datetime.timedelta(days=day)
    print("date",date)
    # number of customers
    m=random.randrange(customer_min,customer_max)
    print("number of customers:",m)
    times=sorted([datetime.time(random.randrange(24),random.randrange(60),random.randrange(60)) for x in range(m)])
    customers=np.random.choice(ids,size=m,replace=True, p=probs)
    for id,t in zip(customers,times):
        dt=datetime.datetime.combine(date,t)
        transaction_type=np.random.choice(["SHOW","DEAL"], p=[0.9,0.1])
        print(transaction_type)
        record_transaction(id,transaction_type,dbname="test_crm.db",dt=dt)
        print(id,t)
