'''
Gromullyang is a working library of classes and functions designed to 
ease the implementation of neural networks in base tensorflow.

Currently, the library contains the capacity to create FFNN, CNN, and RNN
models in limited capacities.

The networks were designed to be used with limit book orders, and includes
pre-processing that falls in line with the "Healthy Markets" dataset.
'''


def setup():
    import tensorflow as tf
    import os
    import numpy as np
    import pandas as pd

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import time

    import matplotlib.pyplot as plt
    import math
    from sklearn.preprocessing import StandardScaler

    import h5py


class preprocessing:
    def reader(self, path): 
        # Reading in the data for 'APPL' stock to np.array
        hf = h5py.File(path, 'r') # '../data/price_level_total_view_2017-01-03.h5'
        data = hf.get('AAPL').value
        # Reading data into pandas dataframe
        pddf = pd.DataFrame()
        for i in range(len(data.dtype.names)):
            dummy = [row[i] for row in data]
            pddf[data.dtype.names[i]] = dummy

        return pddf

    def mid_price(self, pddf): # plug in pddf
        # Mid Price Calculation
        best_ask = np.array([y[0] for y in [x[4] for x in pddf['levels']]])
        best_bid = np.array([y[0] for y in [x[5] for x in pddf['levels']]])
        mid_price = (best_ask + best_bid) / 2
        pddf['mid_price'] = mid_price

        # calculating log of mid_price and getting rid of log values of -inf from the data
        pddf['mid_price_log'] = np.log(pddf['mid_price'])
        pddf = pddf[np.isinf(pddf['mid_price_log']) == False]

        return pddf

    def time_groups(self, pddf, interval=1000000):
        # Group by millisecond
        # New grouped referred to as pddf1
        grouped_pddf = pddf.iloc[:, ]
        grouped_pddf['groupid'] = np.ceil((pddf.timestamp - pddf.iloc[0, :].timestamp) / interval).astype(np.int32)
        grouped_pddf = grouped_pddf.groupby('groupid').last()

        return grouped_pddf

    def target_creation(self, pddf, grouped_pddf):
        # Creating target 
        pddf['mid_price_log_shift'] = pddf['mid_price_log'].shift(periods = -1)
        grouped_pddf['mid_price_log_shift'] = grouped_pddf['mid_price_log'].shift(periods = -1)

        pddf['target'] = np.select([(pddf['mid_price_log_shift'] > pddf['mid_price_log']), (pddf['mid_price_log_shift'] == pddf['mid_price_log']), (pddf['mid_price_log_shift'] < pddf['mid_price_log'])], [2, 1, 0])
        grouped_pddf['target'] = np.select([(grouped_pddf['mid_price_log_shift'] > grouped_pddf['mid_price_log']), (grouped_pddf['mid_price_log_shift'] == grouped_pddf['mid_price_log']), (grouped_pddf['mid_price_log_shift'] < grouped_pddf['mid_price_log'])], [2, 1, 0])

        # Subsetting columns
        pddf_1 = pddf[['trade_volume', 'mid_price_log', 'target']]
        grouped_pddf_1 = grouped_pddf[['trade_volume', 'mid_price_log', 'target']]

        return pddf_1 , grouped_pddf_1

    def add_differential_features(self, df, variable_name):
        shift0 = df[variable_name]
        shift1 = df[variable_name].shift(periods = 1)
        shift2 = df[variable_name].shift(periods = 2)
        shift3 = df[variable_name].shift(periods = 3)
        shift4 = df[variable_name].shift(periods = 4)
        shift5 = df[variable_name].shift(periods = 5)
        shift6 = df[variable_name].shift(periods = 6)
        dummy_name = variable_name + '_direction_0_1'
        df[dummy_name] = np.select([(shift0 > shift1), (shift0 == shift1), (shift0 < shift1)], [2, 1, 0])
        dummy_name = variable_name + '_direction_0_2'
        df[dummy_name] = np.select([(shift0 > shift2), (shift0 == shift2), (shift0 < shift2)], [2, 1, 0])
        dummy_name = variable_name + '_direction_0_3'
        df[dummy_name] = np.select([(shift0 > shift3), (shift0 == shift3), (shift0 < shift3)], [2, 1, 0])
        dummy_name = variable_name + '_direction_0_4'
        df[dummy_name] = np.select([(shift0 > shift4), (shift0 == shift4), (shift0 < shift4)], [2, 1, 0])
        dummy_name = variable_name + '_direction_0_5'
        df[dummy_name] = np.select([(shift0 > shift5), (shift0 == shift5), (shift0 < shift5)], [2, 1, 0])
        dummy_name = variable_name + '_direction_0_6'
        df[dummy_name] = np.select([(shift0 > shift6), (shift0 == shift6), (shift0 < shift6)], [2, 1, 0])
        
        return df


    def add_features(self, pddf_1, grouped_pddf_1):
        # creating differential volume
        pddf_1['trade_volume_differential'] = pddf_1['trade_volume'] - pddf_1['trade_volume'].shift(periods = +1)
        grouped_pddf_1['trade_volume_differential'] = grouped_pddf_1['trade_volume'] - grouped_pddf_1['trade_volume'].shift(periods = +1)

        pddf_1 = add_differential_features(add_differential_features(pddf_1, 'mid_price_log'), 'trade_volume_differential')
        grouped_pddf_1 = add_differential_features(add_differential_features(grouped_pddf_1, 'mid_price_log'), 'trade_volume_differential')

        pddf_1 = pddf_1[np.isnan(pddf_1['trade_volume_differential']) == False]
        grouped_pddf_1 = grouped_pddf_1[np.isnan(grouped_pddf_1['trade_volume_differential']) == False]

        return pddf_1, grouped_pddf_1


    def saver(self, object, file_name):
        # Pickling
        object.to_pickle('../data/price_level_total_view_2017-01-03_AAPL_{0}'.format(file_name))


class FFNN:
    #something
    def startFFNN()
        pass()
class CNN:
    #something
    def startCNN()
        pass()

class RNN:
    #something
    def startRNN()
        pass()