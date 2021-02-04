"""
	This code solves for Question 1
"""

import pandas as pd
import numpy as np
import urllib.request
import os, ssl


# fix the error when using urllib.request.urlretrieve
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

# Download the Trip Record Data
print("######################## Downloading  data... #######################")
url = 'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv'
filename = '/Users/lujicheng/Desktop/capitalone/green_tripdata_2015-09.csv'
urllib.request.urlretrieve(url, filename)

# read the data
green_data = pd.read_csv("green_tripdata_2015-09.csv")
print('Green Taxi data size: ')
print('Number of rows:', green_data.index.size)
print('Number of columns:', green_data.columns.size)
print("########################## Done ! ###################################")
print('')




