
# Start of Project Pre-Processing using CSV

import pandas as pd
import os

os.chdir('C:\\Users\\alxgr\\Documents\\UVA\\DSI\\Spring 2019\\ML\\Project')
df = pd.read_csv("Data/Data_Project-1-Prediction_Book_event_levelII_BAC_book_events_sip-nyse_2016-05-02.csv")
df.head()

def summarize_df(dataframe):
    print("\n\n======= Book Event Type =======\n"+str(df['book_event_type'].value_counts()),"\n"+
        "======= Side =======\n"+str(df['side'].value_counts()), "\n"+
        "======= Order ID =======\n"+str(df['order_id'].value_counts()), "\n"+
        "======= Price =======\n"+str(df['price'].value_counts()),)



summarize_df(df)


sum(df.side =='U'and df.order_id == 0) # trade type T is the same as side U and order_id 0

mod_df = (df[df.book_event_type == 'M'])
trade_df = (df[df.book_event_type == 'T'])

