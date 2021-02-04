"""
	This code solves for Question 3
"""

import pandas as pd
import numpy as np
import datetime as dt
from tabulate import tabulate
import matplotlib.pyplot as plt
from data_preprocess import check_data, feature


# read the data
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

# output mean and median of trip distance by pickup hour
print("########################## Ploting ... ##############################")
fig,ax = plt.subplots(1, 1, figsize=(10, 5))

# use a pivot table to aggregate the trip distance by pickup hour
table = pd.pivot_table(data=green_data, index='Hour', 
	values='Trip_distance',aggfunc={np.mean, np.median}).reset_index()

# rename columns
table.columns = ['Hour','Mean_trip_distance','Median_trip_distance']
table[['Mean_trip_distance','Median_trip_distance']].plot(ax=ax)

plt.xlabel('Hour of day (starting from the midnight)')
plt.ylabel('Miles')
plt.title('Distribution of trip distance by hour of the day.')
plt.xlim([0, 23])
plt.savefig('Question3_mean_median_trip_dist.png')
print('Please close the figure to continue ... \n')
plt.show()

print('=========== Trip distance by hour of the day =========\n')
print(tabulate(table.values.tolist(),["Hour","Mean trip distance","Median trip distance"]))
print("########################## Done ! ###################################")
print('')



# select airport trips
print("######################## Airport trip ###############################")
airport_trips = green_data[(green_data.RateCodeID==2) | (green_data.RateCodeID==3)]
print("Number of trips from/to one of NYC airports: ", airport_trips.shape[0])
print("Average fare of trips from/to NYC airports: ${0:.2f}"
	.format(airport_trips.Fare_amount.mean()))
# print("Average total fare amount of trips from/to NYC airports (before tip): ${0:.2f}"
# 	.format(airport_trips.Total_amount.mean()))


# get the airport trip distance
v1 = airport_trips.Trip_distance
# get the non-airport trip distance
v2 = green_data.loc[~green_data.index.isin(v1.index), 'Trip_distance']

# exculde outliers
v1 = v1[~((v1 - v1.mean()).abs() > 3*v1.std())]
v2 = v2[~((v2 - v2.mean()).abs() > 3*v2.std())] 

# define bins boundaries
bins = np.histogram(v1,density=True)[1]
h1 = np.histogram(v1,bins=bins,density=True)
h2 = np.histogram(v2,bins=bins,density=True)


# plot histogram of airport trips and non-airport trips
fig,ax = plt.subplots(1, 2, figsize = (15,5))
width = 0.5*(bins[1]-bins[0])
ax[0].bar(bins[:-1], h1[0], width=width, alpha=1, color='b')
ax[0].bar(bins[:-1] + width, h2[0], width=width, alpha=1, color='r')
ax[0].legend(['Airport trips','Non-airport trips'], loc='best')
ax[0].set_xlabel('Trip distance (miles)')
ax[0].set_ylabel('Normalized trips count')
ax[0].set_ylim([0, 0.25])
ax[0].set_title('Trip distance distribution.')

# plot histribution of airport trips and non-airport trips by hour of day
airport_trips.Hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
green_data.loc[v2.index, 'Hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
ax[1].set_xlabel('Hour of the day (starting from the midnight)')
ax[1].set_ylabel('Normalized trips count')
ax[1].set_ylim([0, 0.09])
ax[1].set_title('Hourly distribution of trips.')
ax[1].legend(['Airport trips','Non-airport trips'], loc='best')
plt.savefig('Question3_airport_trips.png')
print('Please close the figure to continue ... \n')
plt.show()

print("########################## Done ! ###################################")
print('')

print("######################## Further analysis ###########################")
# plot week_day vs trip_distance
fig,ax = plt.subplots(1, 1, figsize = (9,5))
table2 = pd.pivot_table(data=green_data, index='Week_day', 
	values='Trip_distance',aggfunc={np.mean, np.median}).reset_index()
# rename columns
table2.columns = ['Week_day','Mean_trip_distance','Median_trip_distance']
table2[['Mean_trip_distance','Median_trip_distance']].plot(ax=ax)

plt.xlabel('Day in a week')
plt.ylabel('Miles')
plt.title('Distribution of trip distance by day of the week.')
plt.savefig('Question3_mean_median_weekday.png')
print('Please close the figure to continue ... \n')
plt.show()

# plot week vs trip_distance
fig,ax = plt.subplots(1, 1, figsize = (9,5))
table3 = pd.pivot_table(data=green_data, index='Week', 
	values='Trip_distance',aggfunc={np.mean, np.median}).reset_index()
# rename columns
table3.columns = ['Week','Mean_trip_distance','Median_trip_distance']
table3[['Mean_trip_distance','Median_trip_distance']].plot(ax=ax)

plt.xlabel('Week in a month')
plt.ylabel('Miles')
plt.title('Distribution of trip distance by week of the month.')
plt.savefig('Question3_mean_median_week.png')
print('Please close the figure to continue ... \n')
plt.show()


# plot passenger_count vs trip_distance
fig,ax = plt.subplots(1, 1, figsize = (9,5))
table4 = pd.pivot_table(data=green_data, index='Passenger_count', 
	values='Trip_distance',aggfunc={np.mean, np.median}).reset_index()
# rename columns
table4.columns = ['Passenger_count','Mean_trip_distance','Median_trip_distance']
table4[['Mean_trip_distance','Median_trip_distance']].plot(ax=ax)

plt.xlabel('Passenger count')
plt.ylabel('Miles')
plt.title('Distribution of trip distance with passenger count.')
plt.savefig('Question3_mean_median_passenger.png')
print('Please close the figure to continue ... \n')
plt.show()


print("########################## Done ! ###################################")
print('')








