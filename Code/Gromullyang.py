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


class PreProcessing:


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

class NN_Models:

    def depickler(self,path)
        df = pd.read_pickle(path) '../SYS6016-Final-Project/data/price_level_total_view_2017-01-03_AAPL_grouped_2'
        return df

    def get_DFs(self, df):
        # select columns and scale data
        dfX = df.reset_index().iloc[:,5:]  #np.delete(np.arange(16), [0,2])
        dfX2 = df.reset_index().iloc[:,[2,4]]

        ss = StandardScaler()
        ss.fit(dfX2)
        dfX2 = ss.transform(dfX2)

        dfX2 = pd.DataFrame(data=dfX2, columns= ['mid_price_log', 'trade_volume_differential'])
        dfX = pd.merge(dfX2, dfX, left_index=True, right_index=True)

        dfy = df.reset_index().iloc[:,3]

        # Create train/test sets
        trainarray,labelarray,testarray,testlabelarray = split_data(dfX, dfy, 0.8)
        return trainarray, labelarray, testarray, testlabelarray

        def split_data(self, dfX, dfy, train_frac):

            X = dfX
            y = dfy
            n = X.shape[0]
            cutoff = np.floor(n * train_frac).astype(int) # total - the number you want to test, which here i'm flooring
            #                   (amount you want in training should be 1/10th value the denominator)
            # cutoff

            X_train, X_test = (X.iloc[0:cutoff , :] , X.iloc[cutoff: , :] )
            y_train, y_test = (y.iloc[0:cutoff].values.ravel() , y.iloc[cutoff:].values.ravel() )

            ss = StandardScaler()
            ss.fit(X_train)
            X_train = ss.transform(X_train)
            X_test = ss.transform(X_test)

            return X_train, y_train, X_test, y_test

    def create_datasets(trainarray, labelarray, testarray, testlabelarray, batch_size):
        tf.reset_default_graph()
        
        train_n = trainarray.shape[0]
        test_n = testarray.shape[0]

        with tf.name_scope("dataset"):
            training_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(trainarray, tf.float32),
                        tf.cast(labelarray, tf.int32)
                    )
                ).shuffle(buffer_size=2*train_n).batch(batch_size) # multiply by 2 if using accuracy calc
            )

            test_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(testarray, tf.float32),
                        tf.cast(testlabelarray, tf.int32)
                    )
                )
            ).shuffle(buffer_size=2*test_n).batch(batch_size)

        with tf.name_scope("iterator"):
            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            features, labels = iterator.get_next()
            train_init = iterator.make_initializer(training_dataset) # initializer for train_data
            test_init = iterator.make_initializer(test_dataset) # initializer for train_data

        return features, labels, train_init, test_init, trainarray.shape[1]
        

class FFNN(NN_Models):


    def features_2d_to_3d(self, data, labels, window):
        data_n, data_w = data.shape
        stride1, stride2 = data.strides
        new_len = data_n - window
        data3d = as_strided(data, [new_len , window, data_w], strides=[stride1, stride1, stride2])

        return(data3d, labels[window:])

    def flatten_3d(self, data):
        data_n = data.shape[0]
        new_width = data.shape[1]*data.shape[2]
        
        return np.reshape(data, (data_n, new_width)) # flesh this function out

    def create_datasets(trainarray, labelarray, testarray, testlabelarray, batch_size):
        tf.reset_default_graph()
        
        train_n = trainarray.shape[0]
        test_n = testarray.shape[0]

        with tf.name_scope("dataset"):
            training_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(trainarray, tf.float32),
                        tf.cast(labelarray, tf.int32)
                    )
                ).shuffle(buffer_size=2*train_n).batch(batch_size) # multiply by 2 if using accuracy calc
            )

            test_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(testarray, tf.float32),
                        tf.cast(testlabelarray, tf.int32)
                    )
                )
            ).shuffle(buffer_size=2*test_n).batch(batch_size)

        with tf.name_scope("iterator"):
            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            features, labels = iterator.get_next()
            train_init = iterator.make_initializer(training_dataset) # initializer for train_data
            test_init = iterator.make_initializer(test_dataset) # initializer for train_data

        return features, labels, train_init, test_init, trainarray.shape[1]

    def create_FFNN_model(features, labels, n_inputs, n_outputs=3):
    keep_prob = tf.placeholder(tf.float32)
    
    with tf.name_scope("dnn"):
    
        hidden1 = tf.layers.dense(features, n_inputs, name="hidden1",
                                  activation=tf.nn.elu)#, kernel_regularizer=regularizer1)
        hidden2 = tf.layers.dense(hidden1, n_inputs, name="hidden2",
                                  activation=tf.nn.elu)#, kernel_regularizer=regularizer2)

        #max1 = tf.contrib.layers.maxout(hidden1, n_inputs//2)

    #     dropout1 = tf.nn.dropout(hidden1, keep_prob)
    #     hidden2 = tf.layers.dense(dropout1, n_hidden2, name="hidden2",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer2)
    #     dropout2 = tf.nn.dropout(hidden2, keep_prob)
    #     hidden3 = tf.layers.dense(dropout2, n_hidden3, name="hidden3",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer3)
    #     dropout3 = tf.nn.dropout(hidden3, keep_prob)
    #     hidden4 = tf.layers.dense(dropout3, n_hidden4, name="hidden4",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer4)
    #     dropout4 = tf.nn.dropout(hidden4, keep_prob)
    #     hidden5 = tf.layers.dense(dropout4, n_hidden5, name="hidden5",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer4)
    #     dropout5 = tf.nn.dropout(hidden5, keep_prob)
    #     hidden6 = tf.layers.dense(dropout5, n_hidden5, name="hidden6",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer4)
    #     dropout6 = tf.nn.dropout(hidden6, keep_prob)
    #     hidden7 = tf.layers.dense(dropout6, n_hidden5, name="hidden7",
    #                               activation=tf.nn.relu)#, kernel_regularizer=regularizer4)
    #     dropout7 = tf.nn.dropout(hidden7, keep_prob)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")#, kernel_regularizer=regularizer5)
    

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
    #     l2_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    #     loss += l2_loss
        loss_summary = tf.summary.scalar('log_loss', loss)

    learning_rate = 0.002

    with tf.name_scope("train"):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope("predict"):
        output = tf.nn.softmax(logits)

    merged = tf.summary.merge_all()
    
    return training_op, output, loss, accuracy


    def data_prep(trainarray, labelarray, testarray, testlabelarray):
        # Create train/test sets
        trainarray,labelarray,testarray,testlabelarray = split_data(df, 0.8)
        # transform data set to use data windows
        from numpy.lib.stride_tricks import as_strided
        trainarray, labelarray = features_2d_to_3d(np.array(trainarray), np.array(labelarray), 5)
        testarray, testlabelarray = features_2d_to_3d(np.array(testarray), np.array(testlabelarray), 5)
        trainarray = flatten_3d(trainarray)
        testarray = flatten_3d(testarray)

        # Create datasets & model

        n_inputs = trainarray.shape[1]

        features, labels, tr_init, te_init, n_inputs = create_datasets(trainarray, labelarray, testarray, testlabelarray, batch_size)
        training_op, output, loss, accuracy = create_model(features, labels, n_inputs, n_outputs=3)
        
        return features, labels, tr_init, te_init, n_inputs, training_op, output, loss, accuracy


    def get_hyperparameters(self, train_array, batch_size=100):
        train_n = trainarray.shape[0]
        test_n = testarray.shape[0]
        n_batches = train_n // batch_size
        n_batches_test = test_n // batch_size

        return train_n, test_n, height, width, n_inputs, batch_size, n_batches, n_batches_test

    def run_FFNN(self, n_epochs=10, run_name='FFNN_', init, tr_init, te_init, n_batches, training_op, train_n, test_n):

        columns = ['t-plus', 'loss', 'accuracy', 'test_loss', 'test_accuracy']
        summaries = pd.DataFrame(np.zeros([n_epochs,5], dtype=float), columns=columns)

        with tf.Session() as sess:
            start_time = time.time()
            sess.run(tf.global_variables_initializer())
            tot_batches_run = 0
            for epoch in range(n_epochs):
                sess.run(tr_init) # drawing samples from train_data
                tot_loss = 0
                for i in range(n_batches):
                    try:
                        _, loss_value = sess.run([training_op, loss]) # , feed_dict={keep_prob : 0.75} # for dropout only
                        tot_loss += loss_value
                    except tf.errors.OutOfRangeError:
                        print("out of range on iter {}".format(i))
                        break
                
                
                # Now gauge training accuracy
                sess.run(tr_init) # drawing samples from test_data
                total_correct_preds = 0
                try:
                    while True:
                        accuracy_batch = sess.run(accuracy) # , feed_dict={keep_prob : 1} # for dropout only
                        total_correct_preds += accuracy_batch
                except tf.errors.OutOfRangeError:
                    pass
                tr_acc = total_correct_preds/train_n
                
                # Now get testing loss
                sess.run(te_init) # drawing samples from test_data
                test_tot_loss = 0
                for i in range(n_batches_test):
                    try:
                        loss_value = sess.run([loss]) # , feed_dict={keep_prob : 0.75} # for dropout only
                        test_tot_loss += loss_value[0]
                    except tf.errors.OutOfRangeError:
                        print("out of range on iter {}".format(i))
                        break
                
                # Now gauge testing accuracy
                sess.run(te_init) # drawing samples from test_data
                total_correct_preds = 0
                try:
                    while True:
                        accuracy_batch = sess.run(accuracy) # , feed_dict={keep_prob : 1} # for dropout only
                        total_correct_preds += accuracy_batch
                except tf.errors.OutOfRangeError:
                    pass
                te_acc = total_correct_preds/test_n

                
                epoch_time = time.time()
                print("Epoch: {}, Train_Loss: {:.4f}, Test_Loss: {:.4f}, Train_Accuracy: {:.4f}, Test_Accuracy: {:.4f}"\
                    .format(epoch, tot_loss, test_tot_loss, tr_acc, te_acc))
                cum_time = epoch_time - start_time
                summaries.iloc[epoch,:] = cum_time, tot_loss, test_tot_loss, tr_acc, te_acc
            
                return


class CNN(NN_Models):

    
    def features_2d_to_3d(self, data, labels, window):
        data_n, data_w = data.shape
        stride1, stride2 = data.strides
        new_len = data_n - window
        data3d = as_strided(data, [new_len , window, data_w], strides=[stride1, stride1, stride2])

        return(data3d, labels[window:])

    def flatten_3d(self, data):
        data_n = data.shape[0]
        new_width = data.shape[1]*data.shape[2]
        
        return np.reshape(data, (data_n, new_width)) # flesh this function out
        
    def create_datasets(self, trainarray, labelarray, testarray, testlabelarray, batch_size):
        tf.reset_default_graph()
        
        train_n = trainarray.shape[0]
        test_n = testarray.shape[0]

        with tf.name_scope("dataset"):
            training_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(trainarray, tf.float32),
                        tf.cast(labelarray, tf.int32)
                    )
                ).shuffle(buffer_size=2*train_n).batch(batch_size) # multiply by 2 if using accuracy calc
            )

            test_dataset = (
                tf.data.Dataset.from_tensor_slices(
                    (
                        tf.cast(testarray, tf.float32),
                        tf.cast(testlabelarray, tf.int32)
                    )
                )
            ).shuffle(buffer_size=2*test_n).batch(batch_size)

        with tf.name_scope("iterator"):
            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            features, labels = iterator.get_next()
            train_init = iterator.make_initializer(training_dataset) # initializer for train_data
            test_init = iterator.make_initializer(test_dataset) # initializer for train_data

        return features, labels, train_init, test_init, train_n, test_n

    def create_model(self, features, labels, height, width, pool3_dropout_rate=.25, fc1_dropout_rate=.5):
    
        conv1_fmaps, conv1_ksize, conv1_stride, conv1_pad = 32, 2, 1, "SAME"
        conv2_fmaps, conv2_ksize, conv2_stride, conv2_pad = 64, 2, 1, "SAME"

        # Define a pooling layer
        pool3_dropout_rate = pool3_dropout_rate
        pool3_fmaps = conv2_fmaps

        # Define a fully connected layer
        n_fc1 = 128
        fc1_dropout_rate = fc1_dropout_rate

        # Output
        n_outputs = 3

        #tf.reset_default_graph() # this is called instead in Dataset creation

        # Step 2: Set up placeholders for input data
        with tf.name_scope("inputs"):
            X_reshaped = features
            y = labels
            training = tf.placeholder_with_default(False, shape=[], name='training')

        # Step 3: Set up the two convolutional layers using tf.layers.conv2d
        # Hint: arguments would include the parameters defined above, and what activation function to use
        conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, 
                                strides=(conv1_stride), padding=conv1_pad, activation=tf.nn.relu,
                                name="conv1")
        conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, 
                                strides=(conv2_stride), padding=conv2_pad, activation=tf.nn.relu,
                                name="conv2")

        # Step 4: Set up the pooling layer with dropout using tf.nn.max_pool

        #4-d tensor is [batch, height, width, channel] -- no one does pooling across the batch apparently, so keep that 1 always
        with tf.name_scope("pool3"):
            pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool")    
            pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * height//2 * width//2])
            pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training)

        # Step 5: Set up the fully connected layer using tf.layers.dense
        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
            fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

        # Step 6: Calculate final output from the output of the fully connected layer
        with tf.name_scope("output"):
            # Hint: "Y_proba" is the softmax output of the "logits"
            logits = tf.layers.dense(fc1, n_outputs, name="output")
            Y_proba = tf.nn.softmax(logits, name="Y_proba")

        # Step 5: Define the optimizer; taking as input (learning_rate) and (loss)
        with tf.name_scope("train"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)

        # Step 6: Define the evaluation metric
        with tf.name_scope("eval"):
            correct, accuracy = None, None
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))

        # Step 7: Initiate
        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
        return training_op, loss, accuracy, init, saver




    def windower(self, trainarray, labelarray, testarray, testlabelarray)
        # transform data set to use data windows
        from numpy.lib.stride_tricks import as_strided

        window_size = 6
        # if window_size isn't even, tf.reshape throws an error down below after pooling
        trainarray, labelarray = features_2d_to_3d(np.array(trainarray), np.array(labelarray), window_size)
        testarray, testlabelarray = features_2d_to_3d(np.array(testarray), np.array(testlabelarray), window_size)

        return trainarray, labelarray, testarray, testlabelarray


    def get_hyperparameters(self, trainarray, batch_size=400):
        train_n = trainarray.shape[0]
        test_n = testarray.shape[0]
        height = trainarray.shape[1]
        width = trainarray.shape[2]
        n_inputs = height*width

        n_batches = train_n // batch_size
        n_batches_test = test_n // batch_size

        return train_n, test_n, height, width, n_inputs, batch_size, n_batches, n_batches_test


    def change_shapes(self, array, height, width): # make sure to call this on both train and test
        array = np.reshape(array, newshape=[-1, height, width, 1])
        # trainarray = np.reshape(trainarray, newshape=[-1, height, width, 1])
        # testarray = np.reshape(testarray, newshape=[-1, height, width, 1])

        return array


    def last_prep (self, trainarray, labelarray, testarray, testlabelarray, batch_size):
        features, labels, tr_init, te_init = create_datasets(trainarray, labelarray, testarray, testlabelarray, batch_size)
        training_op, loss, accuracy, init, saver = create_model(features, labels, height, width)


        return features, labels, tr_init, te_init, training_op, loss, accuracy, init, saver

    def run_CNN(self, n_epochs=10, run_name='CNN_', init, tr_init, te_init, n_batches, training_op, train_n, test_n):
        columns = ['t-plus', 'loss', 'accuracy', 'test_loss', 'test_accuracy']
        summaries = pd.DataFrame(np.zeros([n_epochs,5], dtype=float), columns=columns)

        with tf.Session() as sess:
            start_time = time.time()
            #writer.add_graph(sess.graph)
            sess.run(init)
            tot_batches_run = 0
            for epoch in range(n_epochs):
                sess.run(tr_init) # drawing samples from train_data
                train_loss, train_accuracy = 0, 0
                for i in range(n_batches):
                    try:
                        _, loss_value, acc_value = sess.run([training_op, loss, accuracy]) # , feed_dict={keep_prob : 0.75} # for dropout only
                        train_loss += loss_value
                        train_accuracy += acc_value
                    except tf.errors.OutOfRangeError:
                        print("out of range on iter {}".format(i))
                        break
                train_accuracy = train_accuracy/(train_n)
                        
                # Now get testing loss
                sess.run(te_init) # drawing samples from test_data
                test_loss, test_accuracy = 0, 0
                for i in range(n_batches_test):
                    try:
                        loss_value, acc_value = sess.run([loss, accuracy]) # , feed_dict={keep_prob : 0.75} # for dropout only
                        test_loss += loss_value
                        test_accuracy += acc_value
                    except tf.errors.OutOfRangeError:
                        print("out of range on iter {}".format(i))
                        break
                test_accuracy = test_accuracy/(test_n)
                
                
                # epoch_time = time.time()
                print("Epoch: {}, Train_Loss: {:.4f}, Test_Loss: {:.4f}, Train_Accuracy: {:.4f}, Test_Accuracy: {:.4f}".format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))
                # cum_time = epoch_time - start_time
                # summaries.iloc[epoch,:] = cum_time, train_loss, test_loss, train_accuracy, te_acc


class RNN(NN_Models):


    def initial_reshape(self, labelarray):
        from numpy.lib.stride_tricks import as_strided
        np.reshape(as_strided(labelarray, [labelarray.shape[0]-10, 10], strides=[4,4]), [-1,10])[0]
        return labelarray

    def features_2d_to_3d(data, labels, window):
        data_n, data_w = data.shape
        stride1, stride2 = data.strides
        #l_stride1, l_stride2 = labels.strides
        l_stride1 = labels.strides[0]
        new_len = data_n - window
        data3d = as_strided(data, [new_len, window, data_w], strides=[stride1, stride1, stride2])
        #labels3d = as_strided(labels, [new_len, window, 3], strides=[l_stride1, l_stride1, l_stride2])
        labels3d = as_strided(labels, [new_len, window], strides=[l_stride1, l_stride1])
        return data3d, labels3d

    def flatten_3d(data):
        data_n = data.shape[0]
        new_width = data.shape[1]*data.shape[2]
        
        return np.reshape(data, (data_n, new_width))
        
    def batch_data(data, labels, batch_size, n_steps):
        windowed_x, windowed_y = features_2d_to_3d(data, labels, n_steps)
        
        t_steps = data.shape[0]
        width = data.shape[1]
        n_batches = t_steps // batch_size
        remainder = t_steps - (n_batches * batch_size)
        new_len = t_steps - remainder
        
        windowed_x = windowed_x[:new_len]
        windowed_y = windowed_y[:new_len]
        
        x_batches = np.reshape(windowed_x, [-1, batch_size, n_steps, width])
        y_batches = np.reshape(windowed_y, [-1, batch_size, n_steps])
        y_batches = y_batches[:,:,n_steps-1]

        return x_batches, y_batches, n_batches, new_len


    def create_rnn_model(n_steps, batch_size, n_inputs = 14, n_neurons = 100, n_outputs = 3):

        tf.reset_default_graph()

        # Set up placeholders for input data
        X = tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
        labels = tf.placeholder(tf.int32, [batch_size])
        
        with tf.name_scope("rnn"):
            
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
            outputs, final_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
            
        with tf.name_scope("output"):
            logits = tf.layers.dense(final_state, n_outputs, name="output")
            Y_proba = tf.nn.softmax(logits, name="Y_proba")
            
            # print(logits.shape)
            # print(labels.shape)

        # Define the optimizer; taking as input (learning_rate) and (loss)
        with tf.name_scope("train"):
            learning_rate = 0.001
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(xentropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)

        # Step 6: Define the evaluation metric
        with tf.name_scope("eval"):
            correct = tf.nn.in_top_k(logits, labels, 1)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))

        # Step 7: Initiate
        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
        # Summaries
        with tf.name_scope("summaries"):
            value_ = tf.placeholder(shape=[], name='summary_placeholder', dtype=tf.float32)
            #ta = tf.placeholder(shape=[], name='train_accuracy_p', dtype=tf.float32)
            #vl = tf.placeholder(shape=[], name='val_loss_p', dtype=tf.float32)
            #va = tf.placeholder(shape=[], name='val_accuracy_p', dtype=tf.float32)
            tl = tf.summary.scalar('train_loss', value_)
            ta = tf.summary.scalar('train_accuracy', value_)
            vl = tf.summary.scalar('val_loss', value_)
            va = tf.summary.scalar('val_accuracy', value_)
            #merged = tf.summary.merge_all()
        
        return training_op, loss, accuracy, init, X, labels, Y_proba, value_, tl, ta, vl, va

        def windower(self, trainarray, labelarray, testarray, testlabelarray, batch_size=100, window_size=10):
            from numpy.lib.stride_tricks import as_strided

            x, y, n_batches, n_train = batch_data(trainarray, labelarray, batch_size, window_size)
            x_val, y_val, n_batches_val, n_test  = batch_data(testarray, testlabelarray, batch_size, window_size)

            return x, y, n_batches, n_train, x_val, y_val, n_batches_val, n_test 

        def run_RNN(self, n_epochs=10, run_name='RNN_', init, tr_init, te_init, n_batches, training_op, train_n, test_n):
            training_op, loss, accuracy, init, X, labels, Y_proba, value_, tl, ta, vl, va = create_rnn_model(window_size, batch_size)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                writer = tf.summary.FileWriter('../tf_graphs/', tf.get_default_graph())
                init.run()
                for epoch in range(n_epochs):
                    
                    train_loss, train_accuracy = 0, 0
                    for b in range(n_batches):
                        X_batch, y_batch = x[b], y[b]
                        _, loss_value, acc_value = sess.run([training_op, loss, accuracy], feed_dict={X: X_batch, labels: y_batch})
                        train_loss += loss_value
                        train_accuracy += acc_value
                    train_accuracy = train_accuracy/n_train
                    
                    s = sess.run(ta, feed_dict={value_: train_accuracy})
                    writer.add_summary(s, epoch)
                    s = sess.run(tl, feed_dict={value_: train_loss})
                    writer.add_summary(s, epoch)
                    
                    val_loss, val_accuracy = 0, 0
                    for b in range(n_batches_val):
                        X_batch, y_batch = x_val[b], y_val[b]       
                        loss_value, acc_value = sess.run([loss, accuracy], feed_dict={X: X_batch, labels: y_batch})
                        val_loss += loss_value
                        val_accuracy += acc_value
                    val_accuracy = val_accuracy/n_test

                    s = sess.run(va, feed_dict={value_: val_accuracy})
                    writer.add_summary(s, epoch)
                    s = sess.run(vl, feed_dict={value_: val_loss})
                    writer.add_summary(s, epoch)
                    
                    print("Epoch: {}, Train acc: {:.4f}, Train loss: {:.4f}, Val acc: {:.4f}, Val loss: {:.4f}".format(epoch, train_accuracy, train_loss, val_accuracy, val_loss))

                saver.save(sess, "./my_time_series_model")
                writer.close()

                with tf.Session() as sess:                          
                saver.restore(sess, "./my_time_series_model")

                X_new = x_val
                y_pred = sess.run(Y_proba, feed_dict={X: X_new[0]})