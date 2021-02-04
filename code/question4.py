"""
	This code solves for Question 4
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from data_preprocess import check_data, feature
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier



def check_tip(data):
	"""
		This function is to check the distribution of tip percentage.
		Input: data --> pd.DateFrame
	"""

	# get the trip data with tip
	# data_tip = data[data.Tip_percentage > 0]

	fig,ax = plt.subplots(1, 1, figsize=(9,5))
	data.Tip_percentage.hist(bins = 20, density=True, ax=ax)
	ax.set_xlabel('Tip percentage (%)')
	ax.set_ylabel('Normalized count')
	ax.set_title('Tip percentage distribution (all trips)')

	# data_tip.Tip_percentage.hist(bins = 20, density=True, ax=ax[1])
	# ax[1].set_xlabel('Tip percentage (%)')
	# ax[1].set_ylabel('Normalized count')
	# ax[1].set_title('Tip percentage distribution (without non-tip trips)')
	plt.savefig('Question4_tip_check.png')
	print('Please close the figure to continue ... \n')
	plt.show()

	return


def model_train(model, train_data, x_params, target, score_method, do_cv=True, cv_folds=5):
	"""
		This function is to train the model.
		Inputs: model -> estimator
				train_data -> pd.DateFrame
				x_params -> name of the parameters used to train, e.g. ['Payment_type', 'Hour']
				target -> name of the target parameters, e.g. ['Tip_percentage']
				score_method -> 'roc_auc' or 'neg_mean_squared_error'
				do_cv -> perform cross validation if true
				cv_folds -> degree of cross validation
		Output: train_model
	"""

	# train the model on the given dataset
	model.fit(train_data[x_params], train_data[target])
	train_prediction = model.predict(train_data[x_params])
	if (score_method == 'roc_auc'):
		train_data_pred_prob = model.predict_proba(train_data[x_params])[:,1]

	# perform cross validation
	if do_cv:
		cv_score = cross_val_score(model, train_data[x_params], train_data[target], cv=cv_folds, scoring=score_method)
		print("Printing model report...")
		if score_method == 'roc_auc':
			print("Training accuracy: {0:.2f}%".format(metrics.accuracy_score(train_data[target].values, train_prediction)*100))
			print("Training roc-auc score: {0:.5f}".format(metrics.roc_auc_score(train_data[target], train_data_pred_prob)))
		if score_method == 'neg_mean_squared_error':
			# print("Training accuracy: {0:.2f}%".format(metrics.accuracy_score(train_data[target].values, train_prediction)*100))
			print("Training mse: ", metrics.mean_squared_error(train_data[target].values, train_prediction))
		print("Cross-validation score => ")
		print("Mean: %.7g | Std: %.7g | Max: %.7g | Min: %.7g" % (np.mean(cv_score), np.std(cv_score), np.max(cv_score), np.min(cv_score)))

	return


def model_tune(model, train_data, params, x_params, target, score_method, cv_folds=5):
	"""
		This function is to tune the model parameters using exhaustive grid search.
		Inputs: model -> classifier or regressor
				params -> dict, params to be tuned
				train_data -> pd.DateFrame
				x_params -> name of the parameters used to train, e.g. ['Payment_type', 'Hour']
				target -> name of the target parameters, e.g. ['Tip_percentage']
				score_method -> 'roc_auc' or 'neg_mean_squared_error'
				cv_folds -> degree of cross validation
		Output: gs_model -> use its best_estimator_ to train
	"""
	gs_model = GridSearchCV(estimator=model, param_grid=params, scoring=score_method, n_jobs=2, iid=False, cv=cv_folds)
	gs_model.fit(train_data[x_params], train_data[target])
	return gs_model


def predict_tip_percentage(data, classifier, regressor, cls_params, rf_params):
	"""
		This is the predictive model for predicting tip percentage.
		Input: data --> pd.DataFrame
				classifier --> 
				regressor -->
				cls_params --> name of the parameters input to classifier, e.g. ['Payment_type', 'Hour']
				rf_params --> name of the parameters input to regressor
		Output: tip percentage
	"""
	c = classifier.best_estimator_.predict(data[cls_params])
	pred = c*regressor.best_estimator_.predict(data[rf_params])

	return pred





if __name__ == '__main__':
	print("##################### Loading the raw data... #######################")

	raw_data = pd.read_csv("green_tripdata_2015-09.csv")

	print('Green Taxi data size: ')
	print('Number of rows:', raw_data.index.size)
	print('Number of columns:', raw_data.columns.size)
	print("########################## Done ! ###################################")
	print('')

	# check data
	print("######################## Checking data ... ##########################")
	green_data = check_data(raw_data)
	print("########################## Done ! ###################################")
	print('')
	
	# add features to data
	print("####################### Add features to data ...#####################")
	green_data = feature(green_data)
	print("########################## Done ! ###################################")
	print('')

	print("#################### Check tip distribution ... #####################")
	check_tip(green_data)
	
	print("########################## Done ! ###################################")
	print('')


	print("#################### Training the classifier... #####################")
	# training data
	cls_train_data = green_data.loc[np.random.choice(green_data.index, size=100000, replace=False)]

	cls_target = 'Has_tip'
	cls_x = ['Payment_type', 'Total_amount', 'Trip_duration', 'Speed_mph']
	cls_params = {'n_estimators':range(30,150,20)}

	# initialize the RF classifier
	model_classifier = RandomForestClassifier()

	# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	# 	print(train_data[predictors])

	# tune model parameters
	gs_classifier = model_tune(model_classifier, cls_train_data, cls_params, cls_x, cls_target, 'roc_auc')
	print(gs_classifier.best_params_, gs_classifier.best_score_)

	print("start training...")
	model_train(gs_classifier.best_estimator_, cls_train_data, cls_x, cls_target, 'roc_auc')

	print("######################## Training Done ! ############################")
	print('')

	print("###################### Testing classifier... ########################")
	# tesing data
	cls_test_index = green_data.index[~green_data.index.isin(cls_train_data.index)]
	cls_test_data = green_data.loc[np.random.choice(cls_test_index, size=100000, replace=False)]

	classifier_pred = gs_classifier.best_estimator_.predict(cls_test_data[cls_x])
	test_pred_prob = gs_classifier.best_estimator_.predict_proba(cls_test_data[cls_x])[:,1]
	print("Classifier test accuracy: {0:.2f}%".format(metrics.accuracy_score(cls_test_data[cls_target].values, classifier_pred)*100))
	print("Test roc-auc score: {0:.5f}".format(metrics.roc_auc_score(cls_test_data.Has_tip, test_pred_prob)))
	print("######################## Testing Done ! #############################")
	print('')

	print("####################### Training the regressor ...###################")
	rf_data = green_data[green_data.Tip_percentage > 0]
	rf_train_data = rf_data.loc[np.random.choice(rf_data.index, size=100000, replace=False)]
	rf_target = 'Tip_percentage'
	rf_x = ['Payment_type', 'Total_amount', 'Trip_duration', 'Speed_mph']

	params_rf = {'n_estimators':range(50,200,20)}
	# initialize the RF regressor
	model_rf = RandomForestRegressor(100)
	# tune the model parameters
	gs_rf = model_tune(model_rf, rf_train_data, params_rf, rf_x, rf_target, 'neg_mean_squared_error')
	print(gs_rf.best_params_, gs_rf.best_score_)

	model_train(gs_rf.best_estimator_, rf_train_data, rf_x, rf_target, 'neg_mean_squared_error')
	print("######################## Training Done ! ############################")
	print('')

	print("###################### Testing the regressor ... ####################")
	rf_test_index = rf_data.index[~rf_data.index.isin(rf_train_data.index)]
	rf_test_data = rf_data.loc[np.random.choice(rf_test_index, size=100000, replace=False)]

	rf_pred = gs_rf.best_estimator_.predict(rf_test_data[rf_x])
	# print("rf model test accuracy: {0:.2f}%".format(metrics.accuracy_score(rf_test_data.Tip_percentage, rf_pred)*100))
	print('rf model test mse:', metrics.mean_squared_error(rf_test_data.Tip_percentage, rf_pred))
	print('rf model r2 score:', metrics.r2_score(rf_test_data.Tip_percentage, rf_pred))
	print("######################## Testing Done ! #############################")
	print('')

	print("############ Evaluating the final predictive model... ###############")
	test_data = green_data.loc[np.random.choice(green_data.index,size = 100000,replace=False)]
	pred_tip_perc = predict_tip_percentage(test_data, gs_classifier, gs_rf, cls_x, rf_x)

	# print("Predictive model test accuracy: {0:.2f}%".format(metrics.accuracy_score(test_data.Tip_percentage, pred_tip_perc)*100))
	print("Predictive model test mse:", metrics.mean_squared_error(test_data.Tip_percentage, pred_tip_perc))
	print('Predictive model test r2 score:', metrics.r2_score(test_data.Tip_percentage, pred_tip_perc))
	data_copy = test_data.copy()
	data_copy['tip_perc_predict'] = pred_tip_perc
	data_copy['residual'] = data_copy.Tip_percentage - pred_tip_perc
	fig,ax = plt.subplots(1, 1, figsize=(9,5))
	data_copy.residual.hist(bins=20, ax=ax)
	ax.set_xlabel('residual')
	ax.set_ylabel('count')
	ax.set_yscale('log')
	ax.set_title('Histogram of residuals')
	plt.savefig('Question4_residual.png')
	print('Please close the figure to continue ... \n')
	plt.show()
	print("########################## Done ! ###################################")
	print('')





























