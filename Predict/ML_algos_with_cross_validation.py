import GPy
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import Ridge
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from pandas import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
import os

#This class contains methods for different machine learning algorithms
#which takes train and test data and return the prediction errors (absolute error) for test set.
#Each method makes a grid search on the probable hyperparameters settings for the corresponding algorithm.
#Using cross-validation, it determines an optimum hyperparameter setting and then report prediction error
#on the test set using that setting of the hyperparameters.
class MLPredsByCV():
	def __init__(self, cross_validation_split_no = 5):
        self.cv_split_no = cross_validation_split_no
        
        
    def SVR_CV(self, trainX, testX, trainY, testY):
		C_vals = [1.0, 10.0, 100.0, 500.0, 1000.0]
		inverse_gamma_vals = [1.0, 10.0, 20.0, 40.0, 80.0, 200.0]
		epsilon_vals = [0.001, 0.01, 0.1]
		cv_errors = np.empty([len(C_vals)*len(inverse_gamma_vals)*len(epsilon_vals), 4])
		i = 0
		for c in C_vals:
			for g in inverse_gamma_vals:
				for e in epsilon_vals:
					errors = np.empty([self.cv_split_no, 1])
					kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
					j = 0
					for train_indices, validation_indices in kf.split(trainX):
						training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
						training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
						regr = SVR(C=c, gamma=1.0/g, kernel='rbf', epsilon=e)
						regr.fit(training_set_X, training_set_Y)
						predY = regr.predict(validation_set_X)
						errorY = np.absolute(predY - validation_set_Y)
						errors[j] = np.mean(errorY)
						j = j + 1
					cv_errors[i,:] = c, g, e, np.mean(errors)
					i = i + 1
		C_opt, g_opt, eps_opt, _ = cv_errors[np.argmin(cv_errors[:, 3]), :]
		regr = SVR(C=C_opt, gamma=1.0/g_opt, kernel='rbf', epsilon=eps_opt)
		regr.fit(trainX, trainY)
		predY = regr.predict(testX)
		err_on_opt_params = np.absolute(predY - testY)                
		return err_on_opt_params


	def KRR_CV(self, trainX, testX, trainY, testY):
		kernel_vals = ['rbf', 'laplacian']
		kernel_indices = [0,1]
		inverse_gamma_vals = [1.0, 10.0, 20.0, 40.0, 80.0]
		alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
		cv_errors = np.empty([len(kernel_vals)*len(inverse_gamma_vals)*len(alpha_vals), 4])
		i = 0
		for kern in kernel_vals:
			for g in inverse_gamma_vals:
				for a in alpha_vals:
					errors = np.empty([self.cv_split_no, 1])
					kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
					j = 0
					for train_indices, validation_indices in kf.split(trainX):
						training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
						training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
						regr = KernelRidge(alpha=a, gamma=1.0/g, kernel=kern)
						regr.fit(training_set_X, training_set_Y)
						predY = regr.predict(validation_set_X)
						errorY = np.absolute(predY - validation_set_Y)
						errors[j] = np.mean(errorY)
						j = j + 1
					cv_errors[i,:] = kernel_indices[kernel_vals.index(kern)], g, a, np.mean(errors)
					i = i + 1
		k_opt, g_opt, a_opt, _ = cv_errors[np.argmin(cv_errors[:, 3]), :]
		k_opt = kernel_vals[kernel_indices.index(k_opt)]
		regr = KernelRidge(alpha=a_opt, gamma=1.0/g_opt, kernel=k_opt)
		regr.fit(trainX, trainY)
		predY = regr.predict(testX)
		err_on_opt_params = np.absolute(predY - testY)                 
		return err_on_opt_params



	def Ridge_CV(self, trainX, testX, trainY, testY):
		alpha_vals = [0.001, 0.01, 0.1, 1.0]
		cv_errors = np.empty([len(alpha_vals), 2])
		i = 0
		for a in alpha_vals:        
			errors = np.empty([self.cv_split_no, 1])
			kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
			j = 0
			for train_indices, validation_indices in kf.split(trainX):
				training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
				training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
				regr = Ridge(alpha=a)
				regr.fit(training_set_X, training_set_Y)
				predY = regr.predict(validation_set_X)
				errorY = np.absolute(predY - validation_set_Y)
				errors[j] = np.mean(errorY)
				j = j + 1
			cv_errors[i,:] = a, np.mean(errors)
			i = i + 1
		a_opt, _ = cv_errors[np.argmin(cv_errors[:, 1]), :]
		regr = Ridge(alpha=a_opt)
		regr.fit(trainX, trainY)
		predY = regr.predict(testX)
		err_on_opt_params = np.absolute(predY - testY)                
		return err_on_opt_params


	def Lasso_CV(self, trainX, testX, trainY, testY):
		alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
		cv_errors = np.empty([len(alpha_vals), 2])
		i = 0
		for a in alpha_vals:        
			errors = np.empty([self.cv_split_no, 1])
			kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
			j = 0
			for train_indices, validation_indices in kf.split(trainX):
				training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
				training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
				regr = Ridge(alpha=a)
				regr.fit(training_set_X, training_set_Y)
				predY = regr.predict(validation_set_X)
				errorY = np.absolute(predY - validation_set_Y)
				errors[j] = np.mean(errorY)
				j = j + 1
			cv_errors[i,:] = a, np.mean(errors)
			i = i + 1
		a_opt, _ = cv_errors[np.argmin(cv_errors[:, 1]), :]
		regr = Ridge(alpha=a_opt)
		regr.fit(trainX, trainY)
		predY = regr.predict(testX)
		err_on_opt_params = np.absolute(predY - testY)                 
		return err_on_opt_params


	def Elastic_CV(self, trainX, testX, trainY, testY):
		alpha_vals = [0.0001, 0.001, 0.01, 0.1, 1.0]
		l1ratio_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		cv_errors = np.empty([len(alpha_vals)*len(l1ratio_vals), 3])
		i = 0
		for a in alpha_vals:    
			for l in l1ratio_vals:
				errors = np.empty([self.cv_split_no, 1])
				kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
				j = 0
				for train_indices, validation_indices in kf.split(trainX):
					training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
					training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
					regr = linear_model.ElasticNet(alpha=a, l1_ratio=l)
					regr.fit(training_set_X, training_set_Y)
					predY = regr.predict(validation_set_X)
					errorY = np.absolute(predY - validation_set_Y)
					errors[j] = np.mean(errorY)
					j = j + 1
				cv_errors[i,:] = a, l, np.mean(errors)
				i = i + 1
		a_opt, l_opt, _ = cv_errors[np.argmin(cv_errors[:, 2]), :]
		regr = linear_model.ElasticNet(alpha=a_opt, l1_ratio=l_opt)
		regr.fit(trainX, trainY)
		predY = regr.predict(testX)
		err_on_opt_params = np.absolute(predY - testY)                 
		return err_on_opt_params


	def Run_GPy(self, trainX, testX, trainY, testY, kern, ard, vr, ls, nv):
		inputdim = trainX.shape[1]
		kernel = None
		m = None
		if kern == 'rbf':
			kernel = GPy.kern.RBF(input_dim=inputdim, variance=vr, lengthscale=ls, ARD=ard)
		elif kern == 'laplacian':
			kernel = GPy.kern.Exponential(input_dim=inputdim, variance=vr, lengthscale=ls, ARD=ard)
		X = trainX
		Y = np.transpose(np.array([trainY]))
		m = GPy.models.GPRegression(X, Y , kernel, noise_var=nv)
		#m.optimize()
		predY,mseY = m.predict(testX, full_cov=False)
		predY = np.transpose(predY)[0,:]
		mseY = np.transpose(mseY)[0,:]
		errorY = np.absolute(predY - testY)
		return errorY, np.sqrt(mseY)
		

	def GP_CV(self, trainX, testX, trainY, testY):
		kernel_vals = ['rbf', 'laplacian']
		kernel_indices = [0,1]
		ard_vals = [True, False]
		ard_indices = [0,1]
		var_vals = [1.0, 10.0, 50.0, 100.0, 250.0]
		lscale_vals = [1.0, 10.0, 40.0, 80.0, 200.0]
		noise_vals = [0.0001, 0.001, 0.01, 0.1]
		cv_errors = np.empty([len(kernel_vals)*len(ard_vals)*len(var_vals)*len(lscale_vals)*len(noise_vals), 6])
		i = 0
		for kern in kernel_vals:
			for ard in ard_vals: 
				for vr in var_vals:
					for ls in lscale_vals:
						for nv in noise_vals:
							#print(i)
							try:
								errors = np.empty([self.cv_split_no, 1])
								kf = KFold(n_splits=self.cv_split_no, random_state=30, shuffle=True)
								j = 0
								for train_indices, validation_indices in kf.split(trainX):
									training_set_X, validation_set_X = trainX[train_indices], trainX[validation_indices]
									training_set_Y, validation_set_Y = trainY[train_indices], trainY[validation_indices]
									errvals, _ = self.Run_GPy(training_set_X, validation_set_X, training_set_Y, validation_set_Y, 
														kern, ard, vr, ls, nv)
									errors[j] = np.mean(errvals)
									j = j + 1
								cv_errors[i,:] = kernel_indices[kernel_vals.index(kern)], ard_indices[ard_vals.index(ard)]\
													, vr, ls, nv ,np.mean(errors)
								i = i + 1
							except Exception as e:
								print(str(e))
								print('error occured for ', kern, ard, vr, ls, nv)
								raise
		k_opt, a_opt, v_opt, l_opt, n_opt, _ = cv_errors[np.argmin(cv_errors[:, 5]), :]
		k_opt = kernel_vals[kernel_indices.index(k_opt)]
		a_opt = ard_vals[ard_indices.index(a_opt)]
		err_on_opt_params, std_mean_on_opt_params = self.Run_GPy(trainX, testX, trainY, testY, k_opt, a_opt, v_opt, l_opt, n_opt)
		return err_on_opt_params, std_mean_on_opt_params


#This method takes input data and list of ML algorithms and runs them repeatedly (after random shuffling) to get an unbiased 
#estimate of the prediction errors.
#param list:
#outfile: output file path
#case, subcase: these are passed so that if a client code runs this for many types of cases and subcases and then wants to compile
#               the results for all of them, the case and subcase printed as first two columns will help to identify the row.
#dataX: 2D array of shape (no. of total data points, no. of features)	
#dataY: 1D array of shape (no. of total data points,)
#ML_algos: list of machine learning algorithms to run. Possible values: 'ridge','lasso','elastic','krr','svr','gp' corresponding to
#          Ridge Regression, Lasso, Elastic, Kernel Ridge Regression, Support Vector Regression and Gaussian Process, respectively.
#repetitionCount: number of times each algorithm should run repeatedly
#splitIndex: where to split the data into training and testing set.
def RunMLAlgos(outfile, case, subcase, dataX, dataY, ML_algos, repetitionCount, splitIndex):
    df = pd.DataFrame(index = range(len(ML_algos)), columns = ['case', 'subcase', 'algo', 'mean of MAEs', 'std of MAEs',
                                                           'SD of AEs', 'mean of Stds', 'std of Stds'])
	mlrun = MLPredsByCV(cross_validation_split_no = 5)
    j = 0
    for alg in ML_algos:
        print(alg)
        maes = np.zeros(repetitionCount)
        stds = np.zeros(repetitionCount)
        allerrors = []
        for i in range(repetitionCount):
            if i%10 == 0:
                print(i)
            shuffle_order = np.random.permutation(np.arange(dataY.shape[0]))
            X = dataX[shuffle_order]
            Y = dataY[shuffle_order]
            trainX = X[:splitIndex, :]
            testX = X[splitIndex:, :]
            trainY = Y[:splitIndex]
            testY = Y[splitIndex:]
            errors = []
            if alg == 'svr':
                errors = mlrun.SVR_CV(trainX, testX, trainY, testY)
            elif alg == 'krr':
                errors = mlrun.KRR_CV(trainX, testX, trainY, testY)
            elif alg == 'ridge':
                errors = mlrun.Ridge_CV(trainX, testX, trainY, testY)
            elif alg == 'lasso':
                errors = mlrun.Lasso_CV(trainX, testX, trainY, testY)
            elif alg == 'elastic':
                errors = mlrun.Elastic_CV(trainX, testX, trainY, testY)
            elif alg == 'gp':
                errors, stderrs = mlrun.GP_CV(trainX, testX, trainY, testY)
                stds[i] = np.mean(stderrs)
            maes[i] = np.mean(errors)
            allerrors.append(errors)
        df.iloc[j,:] = case, subcase, alg, np.mean(maes), np.std(maes), np.std(allerrors), 'N/A', 'N/A'
        if alg == 'gp':
            df.iloc[j, 6:] = np.mean(stds), np.std(stds)
        j += 1
    df.to_csv(outfile)
        