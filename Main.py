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
from Airport_Trends import *

years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

def listToString(s):
     
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele in s:
        str1 += str(ele) + "_"
 
    # return string
    return str1

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

#avg_table_title = "Airport_CLC_Summary_Table_" + str(datetime.now().date())

avg_table_title = "Airport_Monthly_CLC_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours)

#value_table_title = "Airport_Values_Summary_Table_" + str(datetime.now().date())

value_table_title = "Airport_Monthly_Values_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours)

pd.options.mode.chained_assignment = None

"""
check = 0
twenty_airport_CLC_data = pd.DataFrame({'Year': range(1950,2023)})

slopes = []
r_val = []
p_val = []
PDO_r_val = []
airports = []
airport_count = 0

#for summary VALUES and avg DATA tables

for file in os.listdir('Airport_Data_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in airport_acronyms:
        print(airport_name)
        airport_count += 1
        airports.append(airport_name)
        current_airport_data = graph_airport('Airport_Data_Tables/'+file, years, months, hours)
        slopes.append(current_airport_data[1])
        r_val.append(current_airport_data[2])
        p_val.append(current_airport_data[3])
        PDO_r_val.append(current_airport_data[4])
        twenty_airport_CLC_data[airport_name] = current_airport_data[0]
        print("Graphed " + str(airport_count) + " airport datasets")

twenty_airport_CLC_data.to_csv("CLC_Data/" + avg_table_title + ".csv")

data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

summary_table = pd.DataFrame(data)

summary_table.to_csv("CLC_Data/" + value_table_title + '.csv', sep='\t')

"""
airport_count = 0
slopes = []
r_val = []
p_val = []
PDO_r_val = []
airports = []
airport_count = 0

# 30 year summaries
years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

for era in range(1950, 1983):
    years = range(era, era+30)
    twenty_airport_CLC_data = pd.DataFrame({'Year': years})
    #avg_table_title = "Airport_CLC_Summary_Table_" + str(datetime.now().date())

    avg_table_title = "Airport_Monthly_CLC_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours)

    #value_table_title = "Airport_Values_Summary_Table_" + str(datetime.now().date())

    value_table_title = "Airport_Monthly_Values_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours)

    for file in os.listdir('Airport_Data_Tables'):
        airport_name = file[:file.index(".")]
        elevation = airport_summary["c6"].loc[airport_name]
        if airport_name in airport_acronyms:
            print(airport_name)
            airport_count += 1
            airports.append(airport_name)
            current_airport_data = graph_airport('Airport_Data_Tables/'+file, years, months, hours, elevation)
            slopes.append(current_airport_data[1])
            r_val.append(current_airport_data[2])
            p_val.append(current_airport_data[3])
            PDO_r_val.append(current_airport_data[4])
            twenty_airport_CLC_data[airport_name] = current_airport_data[0]
            print("Graphed " + str(airport_count//30) + " dataset eras")

twenty_airport_CLC_data.to_csv("CLC_Data/" + avg_table_title + ".csv")

data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

summary_table = pd.DataFrame(data)

summary_table.to_csv("CLC_Data/" + value_table_title + '.csv', sep='\t')




