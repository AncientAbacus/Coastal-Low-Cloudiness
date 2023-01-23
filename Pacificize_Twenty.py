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

# gets acronyms for all airports
labels = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary
twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

# function to convert dates and times in given dataset to datetime variables
def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y %m %d %H')

# function to convert given datasets to dataframe
def make_pacific_dataframe(airport_data):
    t = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/' + airport_data, delim_whitespace=True,  names=["year","month","day","hour","cld.frac","cld.base.ft"], parse_dates = {"date" : ["year","month","day","hour",]})
    
    t['date'] = t['date'].apply(date_convert)

    t = t.drop_duplicates(subset=['date'])

    t = t.set_index('date')

    t = t.tz_localize(None)

    # adjust dates from UTC to PST
    t.index = t.index - timedelta(hours=8)

    t.sort_values(by = ['date'])

    return t
    
# loop through given datasets and save dataframes as csv files if each is one of the twenty Pacific Rim airports
check = 0
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Merged_Files'):
    if file[:file.index(".")] in twenty_airport_acronyms:
        make_pacific_dataframe('Merged_Files/' + file).to_csv(r'/Users/ginoangelici/Data_Assistant_Work/Twenty_Tables/' + file[:file.index(".")] + '.csv', sep='\t')
        check += 1
        print("Saved " + str(check) + " airport datasets")
print("Done")


            