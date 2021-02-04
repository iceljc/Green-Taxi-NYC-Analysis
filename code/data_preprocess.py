import pandas as pd
import numpy as np
import datetime as dt

def check_data(raw_data):
	"""
		This function is to check the invalid data
		Input: raw_data, read from csv file
		Output: data --> a pd.DateFrame
	"""
	# copy raw data
	data = raw_data.copy()
	# remove Ehail_fee
	if 'Ehail_fee' in data.columns:
		data.drop('Ehail_fee', axis=1, inplace=True)

	# remove the negative values
	print("replace negative values with their absolute values")
	data.Total_amount = data.Total_amount.abs()
	data.Fare_amount = data.Fare_amount.abs()
	data.improvement_surcharge = data.improvement_surcharge.abs()
	data.Tip_amount = data.Tip_amount.abs()
	data.Tolls_amount = data.Tolls_amount.abs()
	data.MTA_tax = data.MTA_tax.abs()

	if data.Store_and_fwd_flag.dtype.name != 'int64':
		data['Store_and_fwd_flag'] = (data.Store_and_fwd_flag=='Y')*1

	# total amount: the min fare of green taxi is $2.5
	tmp_index = data[(data.Total_amount<2.5)].index
	data.loc[tmp_index, 'Total_amount'] = 2.5

	print("convert time variables to right format ...")
	data['Pickup_datetime'] = data.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
	data['Dropoff_datetime'] = data.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
	
	
	return data



def feature(data):
	
	# copy data
	data = data.copy()

	print("adding time variables ...")
	# get the first week of Sep in 2015
	ref_week = dt.datetime(2015,9,1).isocalendar()[1]
	data['Week'] = data.Pickup_datetime.apply(lambda x:x.isocalendar()[1])-ref_week+1
	data['Week_day'] = data.Pickup_datetime.apply(lambda x:x.isocalendar()[2])
	data['Month_day'] = data.Pickup_datetime.apply(lambda x:x.day)
	data['Hour'] = data.Pickup_datetime.apply(lambda x:x.hour)

	# add trip duration (in min)
	print("adding trip duration ...")
	data['Trip_duration'] = ((data.Dropoff_datetime-data.Pickup_datetime).apply(lambda x:x.total_seconds()/60.0))

	# add avg speed in mph
	print("adding average speed ...")
	data['Speed_mph'] = data.Trip_distance/(data.Trip_duration/60.0)
	tmp_index = data[(data.Speed_mph.isnull()) | (data.Speed_mph>240)].index
	data.loc[tmp_index,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(tmp_index)))

	# add tip percentage
	print("adding tip percentage ...")
	data['Tip_percentage'] = 100 * data.Tip_amount / data.Total_amount
	# add 'Has_tip' to indicate whether the passenger gives tip or not
	data['Has_tip'] = (data.Tip_percentage>0)*1

	return data



