import numpy as np
from pandas import pandas as pd
from enum import Enum
import os
import tensorflow as tf
from MultiFilterNeuralSubnetModel import ChemicalSubnet
from WrapperClasses import BatchNormWrapper, DropoutWrapper, RegularizerWrapper, WeightInitializerWrapper, ActivationFunctions, InitializerTypes
   


class SubnetWrapper():
    
    def __init__(self, maxC, maxO, maxH, num_of_fingerprints, replicationCount, 
                 learn_atomtype_weights, learn_filter_contrib_weights, shareWeightsAcrossAtomTypes,
                 activation, regularizer, hiddenLayer, 
                 initializer, runcnt, dropout = 1.0, learningrate = 0.001, tolerance = 1e-4):
        self.maxC = maxC
        self.maxO = maxO
        self.maxH = maxH
        self.num_of_fingerprints = num_of_fingerprints
        self.replicationCount = replicationCount
        self.learn_atomtype_weights = learn_atomtype_weights
        self.learn_filter_contrib_weights = learn_filter_contrib_weights
        self.shareWeightsAcrossAtomTypes = shareWeightsAcrossAtomTypes
        self.activation = activation
        self.regularizer = regularizer
        self.hiddenLayer = hiddenLayer
        self.initializer = initializer
        self.learningrate = learningrate
        self.runcnt = runcnt
        self.tolerance = tolerance
        if dropout > 0.99:
            self.dropout = DropoutWrapper(False)
        else:
            self.dropout = DropoutWrapper(True, dropout)
    
    def extractDataColsForTF(self, data):
        x = []
        inputlen = self.maxC + self.maxO + self.maxH
        for d in range(inputlen):
            x.append(data[:, d*self.num_of_fingerprints : (d+1)*self.num_of_fingerprints])
        return x
    

    def AppendListOfData(self, dataList):
        if not isinstance(dataList, list):
            if dataList.ndim == 1:
                return dataList.reshape(dataList.shape[0], 1)
            else:
                return dataList
        if len(dataList) == 0:
            print('ERROR! Empty data!!!')
            return None
        for i in range(len(dataList)):
            if dataList[i].ndim == 1:
                dataList[i] = dataList[i].reshape(dataList[i].shape[0], 1)
        appendedData = dataList[0]
        for i in range(1, len(dataList)):
            appendedData = np.vstack((appendedData, dataList[i]))
        return appendedData
    
    def convertForNN(self, X_train, X_validation, X_test, Y_train, Y_validation, Y_test, energyPivot = None):
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(Y_train.shape[0], 1)
        if Y_validation.ndim == 1:
            Y_validation = Y_validation.reshape(Y_validation.shape[0], 1)
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(Y_test.shape[0], 1)
        return self.extractDataColsForTF(X_train), \
                  self.extractDataColsForTF(X_validation),  \
                    self.extractDataColsForTF(X_test), \
                        Y_train, Y_validation, Y_test, energyPivot
                
                
    
    def getInterpolationData(self, encodings, energies, splitIndices):
        encoding = self.AppendListOfData(encodings)
        energy = self.AppendListOfData(energies)
        shuffle_order = np.random.permutation(np.arange(energy.shape[0]))
        energy = energy[shuffle_order]
        encoding = encoding[shuffle_order]
        split1 = splitIndices[0]
        split2 = splitIndices[1]
        X_train, Y_train = encoding[:split1, :], energy[:split1]
        X_validation, Y_validation = encoding[split1:split2, :], energy[split1:split2]
        X_test, Y_test = encoding[split2:, :], energy[split2:]
        return self.convertForNN(X_train, X_validation, X_test, Y_train, Y_validation, Y_test)


    def getExtrapolationData(self, encodingTrains, energyTrains, encodingTests, energyTests, splitIndices):
        encodingTrain = self.AppendListOfData(encodingTrains)
        energyTrain = self.AppendListOfData(energyTrains)
        encodingTest = self.AppendListOfData(encodingTests)
        energyTest = self.AppendListOfData(energyTests)
        shuffle_order = np.random.permutation(np.arange(energyTrain.shape[0]))
        energyTrain = energyTrain[shuffle_order]
        encodingTrain = encodingTrain[shuffle_order]
        split1 = splitIndices[0]
        split2 = splitIndices[1]
        X_train, Y_train = encodingTrain[:split1, :], energyTrain[:split1]
        X_validation, Y_validation = encodingTrain[split1:split2, :], energyTrain[split1:split2]
        return self.convertForNN(X_train, X_validation, encodingTest, Y_train, Y_validation, energyTest)
        

    def prepareSubnetStructures(self, hiddenStructure):
        return [self.num_of_fingerprints] + hiddenStructure + [1]
    
    
    
    def makeInterpolatingPrdictions(self, splitIndices, max_epochs, encoding, energy, 
                                    batchsize = 32, useBatchNorm = False, isVerbose = True):
        maes, stds, errs = [], [], []
        netStruct = self.prepareSubnetStructures(self.hiddenLayer)
        print(netStruct)
        i = 0
        while i < self.runcnt:
            train_x, valid_x, test_x, train_y, valid_y, test_y, _ = self.getInterpolationData( \
                                                                        encoding, energy, splitIndices)
            tf.reset_default_graph()
            net = ChemicalSubnet(self.maxC, self.maxO, self.maxH, self.replicationCount, netStruct, 
                            self.learn_atomtype_weights, self.learn_filter_contrib_weights, self.shareWeightsAcrossAtomTypes,
                            self.activation, self.regularizer, self.dropout,
                            self.initializer, BatchNormWrapper(useBatchNorm), isVerbose = isVerbose)
            
            preds, verr, terr, e = net.RunNN(train_x, valid_x, test_x, train_y, valid_y, \
                                                  learning_rate = self.learningrate, max_epochs = max_epochs, \
                                                  tolerance = self.tolerance)
            aes = abs(preds - test_y)
            print('test set MAE', np.mean(aes))
            print('validation MAE', verr)
            print('training MAE', terr)
            print('saved in epoch', e)
            errs.append(aes)
            maes.append(np.mean(aes))
            stds.append(np.std(aes))
            i += 1
            
        print('Done with mean of MAE of %s' %(np.mean(maes)))
        return np.mean(maes), np.std(maes), np.std(errs)
    
    
    def makeInterpolatingPrdictionsWithEnsemble(self, testSetSize, splitIndices, max_epochs,
                       encoding1, energy1, encoding2, energy2, batchsize = 32, useBatchNorm = False, isVerbose = True):
        maes, stds, errs = [], [], []
        netStruct = self.prepareSubnetStructures(self.hiddenLayer)
        print(netStruct)
        predsDF = pd.DataFrame(index = range(testSetSize), columns = 
                               [i for i in range(self.runcnt)] +
                              ['Mean Pred', 'Std Pred', 'Actual', 'AE'])
        i = 0
        test_y = None
        while i < self.runcnt:
            #Though this is interpolation, but we take a fixed test set and learn ensemble of energies
            #So, getExtrapolationNonSiameseData is called
            train_x, valid_x, test_x, train_y, valid_y, test_y, _ = \
                                                                        self.getExtrapolationData( \
                                                                        encoding1, energy1, encoding2, \
                                                                        energy2, splitIndices)
            tf.reset_default_graph()
            net = ChemicalSubnet(self.maxC, self.maxO, self.maxH, self.replicationCount, netStruct, 
                            self.learn_atomtype_weights, self.learn_filter_contrib_weights, self.shareWeightsAcrossAtomTypes,
                            self.activation, self.regularizer, self.dropout,
                            self.initializer, BatchNormWrapper(useBatchNorm), isVerbose = isVerbose)
            
            preds, verr, terr, e = net.RunNN(train_x, valid_x, test_x, train_y, valid_y, \
                                                  learning_rate = self.learningrate, max_epochs = max_epochs, \
                                                  tolerance = self.tolerance)
            print(preds.shape)
            print(predsDF.shape)
            predsDF.iloc[:, i] = preds
            aes = abs(preds - test_y)
            print('test set MAE', np.mean(aes))
            print('validation MAE', verr)
            print('training MAE', terr)
            print('saved in epoch', e)
            errs.append(aes)
            maes.append(np.mean(aes))
            stds.append(np.std(aes))
            i += 1
            
        predsDF.loc[:, 'Actual'] = test_y
        predictionArray = np.array(predsDF.iloc[:,:(self.runcnt)].values, dtype=np.float32)
        predsDF.loc[:, 'Mean Pred'] = np.mean(predictionArray, axis = 1)
        predsDF.loc[:, 'Std Pred'] = np.std(predictionArray, axis = 1)
        predsDF.loc[:, 'AE'] = abs(predsDF.loc[:,'Actual'] - predsDF.loc[:,'Mean Pred'])
        print('Done with mean of MAE of %s' %(np.mean(maes)))
        print('MAE for Ensemble:', np.mean(predsDF.loc[:, 'AE'].values))
        #predsDF.to_csv(os.path.join(outdir, outputfilename + '.csv'))
        return predsDF.loc[:, 'AE'].values
    
    
    def makeExtrapolatingPrdictions(self,testSetSize, splitIndices, max_epochs,
                       encoding1, energy1, encoding2, energy2, batchsize = 32, useBatchNorm = False, isVerbose = True):
        training_accept_threshold = 1.0 #eV
        maes, stds, errs = [], [], []
        netStruct = self.prepareSubnetStructures(self.hiddenLayer)
        print(netStruct)
        diagnosticOutputs = []
        predsDF = pd.DataFrame(index = range(testSetSize), columns = 
                               [i for i in range(self.runcnt)] +
                              ['Mean Pred', 'Std Pred', 'Actual', 'AE'])
        i = 0
        test_y = None
        max_successive_mistrials = 4*self.runcnt
        num_of_succ_mistrials = 0
        while i < self.runcnt:
            train_x, valid_x, test_x, train_y, valid_y, test_y, _ = \
                                                                        self.getExtrapolationData( \
                                                                        encoding1, energy1, encoding2, \
                                                                        energy2, splitIndices)
            tf.reset_default_graph()
            net = ChemicalSubnet(self.maxC, self.maxO, self.maxH, self.replicationCount, netStruct, 
                            self.learn_atomtype_weights, self.learn_filter_contrib_weights, self.shareWeightsAcrossAtomTypes,
                            self.activation, self.regularizer, self.dropout,
                            self.initializer, BatchNormWrapper(useBatchNorm), isVerbose = isVerbose)
            
            preds, verr, terr, e = net.RunNN(train_x, valid_x, test_x, train_y, valid_y, \
                                                  learning_rate = self.learningrate, max_epochs = max_epochs, \
                                                  tolerance = self.tolerance, batchsize = batchsize)
            if terr > training_accept_threshold:
                num_of_succ_mistrials += 1
                if num_of_succ_mistrials == max_successive_mistrials:
                    print('Training error consistently too high!')
                    raise ValueError('Training error consistently too high!')
                else:
                    continue
            num_of_succ_mistrials = 0
            predsDF.iloc[:, i] = preds
            aes = abs(preds - test_y)
            print('test set MAE', np.mean(aes))
            print('validation MAE', verr)
            print('training MAE', terr)
            print('saved in epoch', e)
            errs.append(aes)
            maes.append(np.mean(aes))
            stds.append(np.std(aes))
            i += 1
            
        predsDF.loc[:, 'Actual'] = test_y
        predictionArray = np.array(predsDF.iloc[:,:(self.runcnt)].values, dtype=np.float32)
        predsDF.loc[:, 'Mean Pred'] = np.mean(predictionArray, axis = 1)
        predsDF.loc[:, 'Std Pred'] = np.std(predictionArray, axis = 1)
        predsDF.loc[:, 'AE'] = abs(predsDF.loc[:,'Actual'] - predsDF.loc[:,'Mean Pred'])
        print('Done with mean of MAE of %s' %(np.mean(maes)))
        print('MAE for Ensemble:', np.mean(predsDF.loc[:, 'AE'].values))
        return np.mean(predsDF.loc[:, 'AE'].values), predsDF.loc[:, 'AE'].values




