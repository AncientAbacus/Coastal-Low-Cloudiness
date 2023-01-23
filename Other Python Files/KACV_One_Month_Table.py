import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pandas as pd
import quandl
import pytz
import math
import os
import glob
import scipy.stats
from scipy import stats
import pandas as pd , numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sma
import seaborn as sns
from Valid import *

airport_to_check = 'PACD'

airport_file = 'Older_Files_TABLES/' + airport_to_check + '.csv'

airport = pd.read_csv('Older_Files_TABLES/' + airport_to_check + '.csv', sep = "\t")

airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

elevation = airport_summary["c6"].loc[airport_to_check]

def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

airport['date'] = airport['date'].apply(date_convert)

airport = airport.set_index('date')

airport = airport[(airport.index.hour == 7) | (airport.index.hour == 10) | (airport.index.hour == 13) | (airport.index.hour == 16)]

airport = airport[(airport.index.month == 9)]

airport = airport[(airport.index.year == 2007)]

valid_airport = airport[(airport["cld.base.ft"] != -999)]
valid_airport = valid_airport[(valid_airport["cld.frac"] != -999.9)] 
valid_airport = valid_airport[(((valid_airport["cld.frac"] > 0.7) & (valid_airport["cld.base.ft"] != 72200)) | ((valid_airport["cld.frac"] < 0.7) & (valid_airport["cld.base.ft"] <= 72200)))]


low_airport = airport[(airport["cld.frac"] >= 0.75) & (((airport["cld.base.ft"]*0.3048)+elevation)<1000)]
low_airport = low_airport[(low_airport["cld.base.ft"] != -999)]
low_airport = low_airport[(low_airport["cld.frac"] != -999.9)] 

airport["cld.base.ft"] = (airport["cld.base.ft"]*0.3048)+elevation

print(airport.to_string())
print("all low observations:")
print(low_airport[low_airport.columns[0]].count())
print("all valid observations:")
print(valid_airport[valid_airport.columns[0]].count())
print("all observations:")
print(airport[airport.columns[0]].count())
print("CLC percent:")
print(low_airport[low_airport.columns[0]].count()/valid_airport[valid_airport.columns[0]].count())



