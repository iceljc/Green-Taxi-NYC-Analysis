"""
	This code solves for Question 5: visualization
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from data_preprocess import check_data, feature



def plot_speed_week(data):

	d = data.copy()
	fig, ax = plt.subplots(1,1,figsize=(9,5))
	d.groupby('Week')['Speed_mph'].mean().plot(kind='line')
	ax.set_xlabel('Week of the month')
	ax.set_ylabel('Average speed (mph)')
	ax.set_title('Average speed vs Week in a month')
	plt.savefig('Question5_speed_week.png')
	print('Please close the figure to continue ... \n')
	plt.show()
	return


def boxplot_speed_week(data):

	data.boxplot('Speed_mph','Week')
	plt.ylim([0,20])
	plt.xlabel('Week of the month')
	plt.ylabel('Average speed (mph)')
	plt.title('Average speed vs Week of the month (boxplot)')
	plt.savefig('Question5_speed_week_boxplot.png')
	print('Please close the figure to continue ... \n')
	plt.show()
	return 


def plot_speed_hour(data):

	d = data.copy()
	fig, ax = plt.subplots(1,1,figsize=(9,5))
	d.groupby('Hour')['Speed_mph'].mean().plot(kind='line')
	ax.set_xlabel('Hour of a day')
	ax.set_xlim([0,23])
	ax.set_ylabel('Average speed (mph)')
	ax.set_title('Average speed vs Hour of a day')
	plt.savefig('Question5_speed_hour.png')
	print('Please close the figure to continue ... \n')
	plt.show()
	return


def boxplot_speed_hour(data):

	data.boxplot('Speed_mph','Hour')
	plt.ylim([5,25]) # cut off outliers
	plt.ylabel('Speed (mph)')
	plt.xlim([0,23])
	plt.title('')
	plt.savefig('Question5_speed_hour_boxplot.png')
	print('Please close the figure to continue ... \n')
	plt.show()
	return


def t_test(data):
	weeks = [1,2,3,4,5]
	p_vals = []
	# run t-test for each pair
	for i in range(len(weeks)): 
		for j in range(len(weeks)):
			p_vals.append((weeks[i], weeks[j],ttest_ind(data[data.Week==weeks[i]].Speed_mph,
				data[data.Week==weeks[j]].Speed_mph, equal_var=False)[1]))
    
	p_values = pd.DataFrame(p_vals, columns=['week_x', 'week_y', 'p_val'])
	
	return p_values


def anova_week():
	weeks = [1,2,3,4,5]
	cmd = "f_oneway("
	for w in weeks:
		cmd+="green_data[green_data.Week=="+str(w)+"].Speed_mph,"
	cmd=cmd[:-1]+")"
	return cmd


def anova_hour():
	hours = range(24)
	cmd = "f_oneway("
	for h in hours:
		cmd+="green_data[green_data.Hour=="+str(h)+"].Speed_mph,"
	cmd=cmd[:-1]+")"
	return cmd




if __name__ == '__main__':
	print("##################### Loading the raw data... #######################")

	raw_data = pd.read_csv("green_tripdata_2015-09.csv")

	print('')
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

	print("############################# Analysis ... ##########################")
	plot_speed_week(green_data)
	boxplot_speed_week(green_data)
	plot_speed_hour(green_data)
	boxplot_speed_hour(green_data)

	print("########################## Done ! ###################################")
	print('')

	print("####################### Statistical test ... ########################")
	
	print("Analysis of week vs speed: ")
	p_values = t_test(green_data)
	print("Performing t-test ==>")
	print("p-values:\n",pd.pivot_table(p_values, index='week_x', columns='week_y', 
		values='p_val').T)
	
	print('')
	print('Performing ANOVA test ==>')
	cmd1 = anova_week()
	print("Anova test result:", eval(cmd1))
	print('')

	print("Analysis of hour vs speed: ")
	print('Performing ANOVA test ==>')
	cmd2 = anova_hour()
	print("Anova test result:", eval(cmd2))
	print("########################## Done ! ###################################")
	print('')


