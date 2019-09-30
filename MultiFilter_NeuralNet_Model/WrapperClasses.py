import numpy as np
import tensorflow as tf
from enum import Enum


class BatchNormWrapper():
    def __init__(self, useBatchNorm = False, offset = 0, scale = 1, var_eps = 1e-3):
        self.useBatchNorm = useBatchNorm
        self.offset = offset
        self.scale = scale
        self.var_eps = var_eps
        
    def performBatchNorm(self, matmul_plus_bias):
        if not self.useBatchNorm:
            return matmul_plus_bias
        batch_mean2, batch_var2 = tf.nn.moments(matmul_plus_bias,[0])
        return tf.nn.batch_normalization(matmul_plus_bias, batch_mean2, batch_var2, self.offset, self.scale, self.var_eps)
    
class DropoutWrapper():
    def __init__(self, useDropout = False, keep_prob = 0.9):
        self.useDropout = useDropout
        self.keep_prob = keep_prob
        
    def performDropout(self, matmul_plus_bias):
        if not self.useDropout:
            return matmul_plus_bias
        return tf.nn.dropout(matmul_plus_bias, self.keep_prob)
    
class RegularizerWrapper():
    def __init__(self, useRegularizer, scale = 0.0):
        self.useRegularizer = useRegularizer
        self.scale = scale
        
    def getRegularizer(self):
        if not self.useRegularizer:
            return None
        return tf.contrib.layers.l2_regularizer(scale=self.scale)
    
    def toStr(self):
        return str(self.scale)
    

ActivationFunctions = Enum('ActivationFunctions', 'relu tanh sigmoid')
    
InitializerTypes = Enum('InitializerTypes','RandomNormal RandomUniform TruncatedNormal GlorotNormal GlorotUniform')

class WeightInitializerWrapper():
    def __init__(self, initializerType, 
                 rand_norm_mean = 0.0, rand_norm_stddev = 1.0, 
                 rand_uniform_min = -1.0, rand_uniform_max = 1.0,
                 trunc_norm_mean = 0.0, trunc_norm_stddev = 1.0):
        self.init = initializerType
        self.rand_norm_mean = rand_norm_mean
        self.rand_norm_stddev = rand_norm_stddev
        self.rand_uniform_min = rand_uniform_min
        self.rand_uniform_max = rand_uniform_max
        self.trunc_norm_mean = trunc_norm_mean
        self.trunc_norm_stddev = trunc_norm_stddev
        
    def getInitializer(self):
        if self.init == InitializerTypes.RandomNormal:
            return tf.random_normal_initializer(mean = self.rand_norm_mean, stddev = self.rand_norm_stddev)
        if self.init == InitializerTypes.RandomUniform:
            return tf.random_uniform_initializer(minval = self.rand_uniform_min, maxval = self.rand_uniform_max)
        if self.init == InitializerTypes.TruncatedNormal:
            return tf.truncated_normal_initializer(mean = self.trunc_norm_mean, stddev = self.trunc_norm_stddev)
        if self.init == InitializerTypes.GlorotNormal:
            return tf.glorot_normal_initializer()
        if self.init == InitializerTypes.GlorotUniform:
            return tf.glorot_uniform_initializer()
        
    def toStr(self):
        if self.init == InitializerTypes.RandomNormal:
            return 'RN' + str(self.rand_norm_stddev)
        if self.init == InitializerTypes.RandomUniform:
            return 'RU' + str(self.rand_uniform_min) + '-' + str(self.rand_uniform_max)
        if self.init == InitializerTypes.TruncatedNormal:
            return 'TN' + str(self.trunc_norm_stddev)
        if self.init == InitializerTypes.GlorotNormal:
            return 'GN'
        if self.init == InitializerTypes.GlorotUniform:
            return 'GU'
        
        
class PrintInfo():
    def __init__(self):
        pass
    
    def printEpochResult(self, epoch, validationError, trainError):
        print("======= AT EPOCH", epoch, " =========") 
        print('train error: ', trainError)
        print('validation error: ', validationError)
        
    def printSavingInfo(self, validationError):
        print('saving lower cost weights')
        print('At validation error: ', validationError)
    
    def printFinalResults(self, trainError, validationError):
        print("Training Done. Printing results...")
        print("Training Error = ", trainError) 
        print("Validation Error = ", validationError) 
    




    




