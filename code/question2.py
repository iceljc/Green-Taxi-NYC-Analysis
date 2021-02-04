"""
	This code solves for Question 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# read the data
print("##################### Loading the raw data... #######################")
green_data = pd.read_csv("green_tripdata_2015-09.csv")

print('Green Taxi data size: ')
print('Number of rows:', green_data.index.size)
print('Number of columns:', green_data.columns.size)
print("########################## Done ! ###################################")
print('')

print("####################### Ploting histogram ... #######################")
# define figure
fig,ax = plt.subplots(1,2,figsize = (15,5)) 

# histogram of trip distance
green_data.Trip_distance.hist(bins=30,ax=ax[0])
ax[0].set_xlabel('Trip Distance (miles)')
ax[0].set_ylabel('Count')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of Trip Distance with outliers')

# get the Trip Distance data
trip_dist = green_data.Trip_distance

# exclude the outliers
trip_dist_filter = trip_dist[~((trip_dist - trip_dist.mean()).abs() > 3*trip_dist.std())]
trip_dist_filter.hist(bins=30, ax=ax[1]) 
ax[1].set_xlabel('Trip Distance (miles)')
ax[1].set_ylabel('Count')
ax[1].set_title('Histogram of Trip Distance without outliers')

# apply a lognormal fit -- use trip distance mean 
scatter, loc, mean = lognorm.fit(green_data.Trip_distance.values, 
	scale=green_data.Trip_distance.mean(), loc=0)
pdf = lognorm.pdf(np.arange(0,12,0.2), scatter, loc, mean)
ax[1].plot(np.arange(0,12,0.2),700000*pdf,'r') 
ax[1].legend(['lognormal fit', 'raw data'])

plt.savefig('Question2_trip_dist_histo.png')
print('Please close the figure to continue ...')
plt.show()
print("########################## Done ! ###################################")



