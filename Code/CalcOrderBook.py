#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:16:50 2019

@author: SM
"""

# define class using heaps to store order book
from heapq import heappush, heappop
#import itertools

from dataclasses import dataclass, field
#from typing import Any

@dataclass(order=True)
class LimitOrder:
    price: int
    quantity: int=field(compare=False)
    order_id: int=field(compare=False)
#    num: int = field(default=0)
        

class OrderBook():
    def __init__(self):
        self.ask_q = []
        self.bid_q = []
        self.ask_order_finder = {}
        self.bid_order_finder = {}
        self.REMOVED = '<removed-task>'      # placeholder for a removed task
#        self.counter = itertools.count()     # unique sequence count
        self.price = -1
        self.best_ask = -1
        self.best_bid = -1
        
        
        # DO I need to include a sequential "count" to the limit orders so that the heap works correctly?
        
    def process_row(self, row):
        'Modifies order book based on contents of order book data set row'
        lo = LimitOrder(price=row['price'], quantity=row['quantity'], order_id=row['order_id'])
        if row['side'] == "A":
            if row['book_event_type'] == "A":
                self.add_ask_order(lo)
            elif row['book_event_type'] == "M":
                self.add_bid_order(lo)
            elif row['book_event_type'] == "C":
                self.remove_ask_order(row['order_id'])
            else:
                print("Error: book_event_type {} not found".format(row['book_event_type']))
        elif row['side'] == "B":
            if row['book_event_type'] == "A":
                self.add_bid_order(lo)
            elif row['book_event_type'] == "M":
                self.add_bid_order(lo)
            elif row['book_event_type'] == "C":
                self.remove_bid_order(row['order_id'])
            else:
                print("Error: book_event_type {} not found".format(row['book_event_type']))
        elif row['side'] == "U":
            if row['book_event_type'] == "T":
                self.calculate_price()
            else:
                print("Error: book_event_type {} not found for side=U".format(row['book_event_type']))
                
                ###FIX may need to do way more here: need to modify quantities in order book heaps and such
                # need to understand data better first, however - how do transactions work in data set?
        else:
            print("Error: Order side {} not found".format(row['side']))
            
        ###FIX may need to return some useful information after each row, e.g. current price, order book depth
        
    def add_ask_order(self, lo):
        'Add a new order or update the priority of an existing order'
        if lo.order_id in (self.ask_order_finder):
            self.remove_ask_order(lo.order_id)
        self.ask_order_finder[lo.order_id] = lo
        heappush(self.ask_q , lo)
        
    def add_bid_order(self, lo):
        'Add a new order or update the priority of an existing order'
        if lo.order_id in self.bid_order_finder:
            self.remove_bid_order(lo.order_id)
        self.bid_order_finder[lo.order_id] = lo
        heappush(self.bid_q, lo)
        
    def remove_ask_order(self, order_id):
        'Mark an existing order as REMOVED.  Raise KeyError if not found.'
        lo = self.ask_order_finder.pop(order_id)
        lo.order_id = self.REMOVED
        
    def remove_bid_order(self, order_id):
        'Mark an existing order as REMOVED.  Raise KeyError if not found.'
        lo = self.bid_order_finder.pop(order_id)
        lo.order_id = self.REMOVED
        
    def calculate_price(self):
        try:
            while self.ask_q:
                lo = heappop(self.ask_q)
                if lo.order_id is not self.REMOVED:
                    del self.ask_order_finder[lo.order_id]
                    self.best_ask = lo.price
        except:
            print("Ask queue empty")
        try:
            while self.bid_q:
                lo = heappop(self.bid_q)
                if lo.order_id is not self.REMOVED:
                    del self.bid_order_finder[lo.order_id]
                    self.best_bid = lo.price
        except:
            print("Bid queue empty")
        if (self.best_ask > -1) and (self.best_bid > -1):
            self.price = (self.best_ask + self.best_bid)/2
        print(self.price)

    def get_price(self):
        return(self.price)
    
        
# Test the parsing
        
# Start of Project Pre-Processing using CSV

import pandas as pd
import os
import numpy as np

#os.chdir('C:\\Users\\alxgr\\Documents\\UVA\\DSI\\Spring 2019\\ML\\Project')
os.chdir("/Users/SM/DSI/classes/spring2019/SYS6016/FinalProject/SYS6016-Final-Project/")

df = pd.read_csv("Data_Project-1-Prediction_Book_event_levelII_BAC_book_events_sip-nyse_2016-05-02.csv")

temp = df.iloc[0:100,:]

ob = OrderBook()    
for index, row in temp.iterrows():
#    print(row)
    ob.process_row(row)
    ob.calculate_price()
    ob.get_price()