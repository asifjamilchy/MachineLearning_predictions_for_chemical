import numpy as np
from pandas import pandas as pd
import tensorflow as tf
from enum import Enum
import os
from WrapperClasses import BatchNormWrapper, DropoutWrapper, RegularizerWrapper, WeightInitializerWrapper, PrintInfo, ActivationFunctions, InitializerTypes





class ChemicalSubnet():
    
    def __init__(self, maxC, maxO, maxH, replicationCount, subnetStructure, 
                 learn_atomtype_weights, learn_filter_contrib_weights, shareWeightsAcrossAtomTypes,
                 activation, regularizer, dropout, initializer, batchNorm, isVerbose = True):
        self.maxC = maxC
        self.maxO = maxO
        self.maxH = maxH
        self.replicationCount = replicationCount
        self.subnetStructure = subnetStructure
        self.numberOfHiddenLayers = len(self.subnetStructure)-2
        self.learn_atomtype_weights = learn_atomtype_weights
        self.learn_filter_contrib_weights = learn_filter_contrib_weights
        self.shareWeightsAcrossAtomTypes = shareWeightsAcrossAtomTypes
        self.activation = activation
        self.isVerbose = isVerbose
        self.initializer = initializer
        self.regularizer = regularizer
        self.batchNorm = batchNorm
        self.dropout = dropout
        
        
    def runActivationFunction(self, matmul):
        if self.activation == ActivationFunctions.tanh:
            return tf.nn.tanh(matmul)
        elif self.activation == ActivationFunctions.sigmoid:
            return tf.nn.sigmoid(matmul)
        return tf.nn.relu(matmul)
    

    def makeSubnetStructure(self):   
        weights = []
        for i in range(self.numberOfHiddenLayers + 1):
            weights.append(tf.get_variable('wt'+str(i), shape=[self.subnetStructure[i], self.subnetStructure[i+1]], 
                           initializer=self.initializer.getInitializer(), regularizer = self.regularizer.getRegularizer()))
        return weights
    
    
    def build_subnet(self, subnet_input, subnetname):
        with tf.variable_scope(subnetname, reuse=True): 
            weights = self.makeSubnetStructure()
            layer_output = [subnet_input]
            for i in range(self.numberOfHiddenLayers):
                matmul_plus_bias =  self.dropout.performDropout(
                                       self.runActivationFunction(
                                        self.batchNorm.performBatchNorm(
                                            tf.matmul(layer_output[i], weights[i])
                                        )
                                       )
                                     )
                layer_output.append(matmul_plus_bias)
            return tf.matmul(layer_output[-1], weights[self.numberOfHiddenLayers])
    
    
    def build_model(self, inputs):
        filter_output = 0.0
        for k in range(self.replicationCount):
            sumC, sumO, sumH = 0, 0, 0
            filterWeight = tf.get_variable('filterWeight'+str(k), shape=[1,1], 
                            initializer=self.initializer.getInitializer(), regularizer = self.regularizer.getRegularizer())
            if not self.learn_filter_contrib_weights:
                filterWeight = 1
            cnet, onet, hnet = 'cnet', 'cnet', 'cnet'
            if not self.shareWeightsAcrossAtomTypes:
                onet = 'onet'
                hnet = 'hnet'
            for i in range(self.maxC):
                sumC += self.build_subnet(inputs[i], cnet + str(k))
            for i in range(self.maxH):
                sumH += self.build_subnet(inputs[self.maxC + i], hnet + str(k))
            for i in range(self.maxO):
                sumO += self.build_subnet(inputs[self.maxC + self.maxH + i], onet + str(k))
            with tf.variable_scope('globwts', reuse=True):
                cwt = tf.get_variable('c_wt', shape=[1,1], 
                                initializer=tf.constant_initializer(1.0))#, regularizer = self.regularizer.getRegularizer())
                owt = tf.get_variable('o_wt', shape=[1,1], 
                                initializer=tf.constant_initializer(1.0))#, regularizer = self.regularizer.getRegularizer())
                hwt = tf.get_variable('h_wt', shape=[1,1], 
                                initializer=tf.constant_initializer(1.0))#, regularizer = self.regularizer.getRegularizer())
                if not self.learn_atomtype_weights:
                    cwt = owt = hwt = 1
                filter_output += ((sumC * cwt) + (sumO * owt) + (sumH * hwt)) * filterWeight #
        return filter_output
    
    
    def getBatchData(self, train_x, train_y, batchsize, batchno):
        tx = []
        for j in range(len(train_x)):
            tx.append((np.array(train_x[j]))[batchno*batchsize: (batchno+1)*batchsize,:])
        ty = train_y[batchno*batchsize: (batchno+1)*batchsize]
        return tx, ty
    
    def getPermutedDataForIteration(self, train_x, train_y):
        tx = []
        shuffle_order = np.random.permutation(np.arange(train_y.shape[0]))
        for j in range(len(train_x)):
            tx.append((np.array(train_x[j]))[shuffle_order])
        ty = train_y[shuffle_order]
        return tx, ty


    def RunNN(self, train_x, valid_x, test_x, train_y, valid_y, learning_rate = 0.01, max_epochs = 10000, tolerance = 1e-4,
             batchsize = 32):
        tf.reset_default_graph()
        
        for i in range(self.replicationCount):
            with tf.variable_scope('cnet'+str(i)): 
                self.makeSubnetStructure() 
            with tf.variable_scope('hnet'+str(i)): 
                self.makeSubnetStructure() 
            with tf.variable_scope('onet'+str(i)): 
                self.makeSubnetStructure() 
                
        with tf.variable_scope('globwts'):
            tf.get_variable('c_wt', shape=[1,1])
            tf.get_variable('o_wt', shape=[1,1])
            tf.get_variable('h_wt', shape=[1,1])
                
        with tf.name_scope("IO"):
            inputarr = []
            inputlen = self.maxC + self.maxO + self.maxH
            for i in range(inputlen):
                inputarr.append(tf.placeholder(tf.float32, [None, self.subnetStructure[0]], name="X"+str(i)))
            outputs = tf.placeholder(tf.float32, [None, 1], name="Yhat")    

        with tf.name_scope("train"):
            yout = self.build_model(inputarr)    
            mae_cost_op = tf.reduce_mean(abs(yout - outputs))
            cost_op = tf.reduce_mean(tf.pow(yout - outputs, 2))
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

        epoch = 0
        current_min_validation_error = 1000.0
        printInfo = PrintInfo()
        if self.isVerbose:
            print( "Beginning Training" )
        init = tf.global_variables_initializer()
        sess = tf.Session() # Create TensorFlow session
        saver = tf.train.Saver()
        save_dir = 'checkpoints/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'best_validation1')
        with sess.as_default():
            sess.run(init)
            traindict = {i:t for i,t in zip(inputarr, train_x)} 
            traindict[outputs] = train_y
            validdict = {i:t for i,t in zip(inputarr, valid_x)} 
            validdict[outputs] = valid_y
            testdict = {i:t for i,t in zip(inputarr, test_x)} 
            savedEpoch = -1
            numbatch = int(train_y.shape[0]/batchsize)
            while True:                
                train_iter_x, train_iter_y = self.getPermutedDataForIteration(train_x, train_y)
                for i in range(numbatch):
                    tx, ty = self.getBatchData(train_iter_x, train_iter_y, batchsize, i)
                    tdict = {mi:mt for mi,mt in zip(inputarr, tx)} 
                    tdict[outputs] = ty
                    sess.run( train_op, feed_dict = tdict)
                    cost_validation = sess.run(mae_cost_op, feed_dict = validdict)
                    if current_min_validation_error - cost_validation > tolerance:
                        if self.isVerbose == True:
                            printInfo.printSavingInfo(cost_validation)
                        saver.save(sess=sess, save_path=save_path)
                        savedEpoch = epoch
                        current_min_validation_error = cost_validation
                        
                if self.isVerbose:
                    printInfo.printEpochResult(epoch, cost_validation, sess.run(mae_cost_op, feed_dict = traindict))
                if epoch > max_epochs :
                    break   
                epoch += 1

            saver.restore(sess=sess, save_path=save_path)
            if self.isVerbose:
                printInfo.printFinalResults(sess.run(cost_op, feed_dict= traindict), sess.run(cost_op, feed_dict= validdict))   
                
            return sess.run(yout, feed_dict=testdict), \
                      sess.run(cost_op, feed_dict= validdict), sess.run(cost_op, feed_dict= traindict), savedEpoch
        
            
            
    




    