import numpy as np
from pandas import pandas as pd
import tensorflow as tf
from enum import Enum
from WrapperClasses import RegularizerWrapper, WeightInitializerWrapper, ActivationFunctions, InitializerTypes
from SubnetWrapper import SubnetWrapper
import os

def readFiles(encodingFile, energyFile, encodingColsToUse, energyColsToUse, spcToRemove, spcAsPivot = None, spcToAdd = None):
    rowsToRemove = []
    rowsAsPivot = []
    energyDF = pd.read_csv(energyFile, skiprows=0, index_col=0)
    for i in range(energyDF.shape[0]):
        if energyDF.loc[i,'species'] in spcToRemove:
            rowsToRemove.append(i)
        if spcToAdd is not None and energyDF.loc[i,'species'] not in spcToAdd:
            rowsToRemove.append(i)
        if spcAsPivot is not None and energyDF.loc[i,'species'] in spcAsPivot:
            rowsAsPivot.append(i)
    encoding = np.loadtxt(encodingFile, delimiter=',', skiprows=1, usecols=encodingColsToUse)
    energy = np.loadtxt(energyFile, delimiter=',', skiprows=1, usecols=energyColsToUse)
    encodingPivot = None
    energyPivot = None
    if len(rowsAsPivot) > 0:
        encodingPivot = encoding[rowsAsPivot,:]
        energyPivot = energy[rowsAsPivot]
    rowsToRemove.extend(rowsAsPivot)
    encoding = np.delete(encoding, rowsToRemove, 0)
    energy = np.delete(energy, rowsToRemove, 0)
    return encoding, energy, encodingPivot, energyPivot





def RunForFiltersAndRuns(baseDir, numOfFP, hiddlayers, testSetSize, filterRange, runsForFilter, runsForEnsemble):
    savepath = os.path.join(baseDir, 'RunResults_FP' + str(numOfFP) + '_hidd' + str(hiddlayers) + '.csv')
    collist = ['FilterCount','RunNo','MAE'] + ['AE'+str(i) for i in range(testSetSize)]
    if not os.path.exists(savepath):
        df = pd.DataFrame(columns = collist)
        df.to_csv(savepath)

    for k in filterRange:
        print('Replication count', str(k+1))
        for i in range(runsForFilter):
            print('RUN', str(i+1))
            df = pd.read_csv(savepath, index_col=0)
            if len(df[(df['FilterCount']==k+1) & (df['RunNo']==i+1)]) > 0:
                print('Already computed. Continuing...')
                continue
            print('Computing...')
            pred = SubnetWrapper(maxC = 4, maxO = 4, maxH = 0, num_of_fingerprints = numOfFP, replicationCount = k+1, 
                    learn_atomtype_weights = True, learn_filter_contrib_weights = True, shareWeightsAcrossAtomTypes = True,
                                 activation = ActivationFunctions.tanh, regularizer = RegularizerWrapper(True, 0.001), 
                                 hiddenLayer = hiddlayers, dropout = 0.9,
                              initializer = WeightInitializerWrapper(InitializerTypes.RandomNormal, rand_norm_stddev = 0.0001), 
                                 learningrate = 0.001, runcnt = runsForEnsemble, tolerance = 1e-3)
            mae, aes_for_this_run = pred.makeExtrapolatingPrdictions(testSetSize = testSetSize, splitIndices = [217,247], 
                                                                     max_epochs = 10000,
                                                                     encoding1 = encodings, energy1 = energies, 
                                                                     encoding2 = enctest, energy2 = engtest, 
                                                                     batchsize = 217, useBatchNorm = False, isVerbose = False)
            print('MAE for current run:', mae)
            df.loc[len(df)] = dict(zip(collist, [k+1, i+1, mae] + aes_for_this_run.tolist()))
            df.to_csv(savepath)
            
            
removeList = ['IM102']# ['IM102','IM74','IM93','s3','s5']
encoding_succ, energy_succ, _, _ = readFiles('Extrapolation/SUCC_8_8_fingerprints_for_subnet_sorted.csv',
                                       'Extrapolation/SUCC_Energies.csv', range(4,68), 4, removeList)
encoding_succ_dcx, energy_succ_dcx, _, _ = readFiles('Extrapolation/SUCC_DCX_8_8_fingerprints_for_subnet_sorted.csv',
                                       'Extrapolation/SUCC_DCX_Energies.csv', range(3,67), 4, removeList)
encodings = [encoding_succ, encoding_succ_dcx]
energies = [energy_succ, energy_succ_dcx]

encoding_prop, energy_prop, _, _ = readFiles('Extrapolation/PAC_8_8_fingerprints_for_subnet_sorted.csv',
                                    'Extrapolation/PAC_Energies.csv', range(3,67), 4, 
                                    ['CH3','CH3O','CH3OH','CO','CO2','COOH','H2O','OH','CH2C','CH3C',], None, None)
encoding_prop_alc, energy_prop_alc, _, _ = readFiles('Extrapolation/PAC_alcohol_8_8_fingerprints_for_subnet_sorted.csv',
                                'Extrapolation/PAC_alcohol_2001_Energies.csv', range(3,67), 4, ['CH3CH2CH3'], None, None)
enctest = [encoding_prop, encoding_prop_alc]
engtest = [energy_prop, energy_prop_alc]

RunForFiltersAndRuns(baseDir = 'Extrapolation', numOfFP = 8, hiddlayers = [10], testSetSize = 29,
                    filterRange = range(8,16), runsForFilter = 10, runsForEnsemble = 8)     