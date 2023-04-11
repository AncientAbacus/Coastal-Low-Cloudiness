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
import warnings
from Airport_CLC_Calculation import *


"""
# Average Tables and Value Tables for each year in given range

#------------------------------------------------------
years = range(1950, 2023)
months = ["May", "June", "July", "August", "September"]
hours = [7, 10, 13, 16]
elevation_def = 1000
#------------------------------------------------------

pd.options.mode.chained_assignment = None


# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

avg_table_title = "Airport_CLC_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + "_Months_" + listToString(months) + "_Hours_" + listToString(hours)

value_table_title = "Airport_Values_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + "_Months_" + listToString(months) + "_Hours_" + listToString(hours)

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
        elevation = elevation = airport_summary["c6"].loc[airport_name]
        print(airport_name)
        airport_count += 1
        airports.append(airport_name)
        current_airport_data = calculate_airport_data('Airport_Data_Tables/'+file, years, months, hours, elevation, elevation_def)
        slopes.append(current_airport_data[1])
        r_val.append(current_airport_data[2])
        p_val.append(current_airport_data[3])
        PDO_r_val.append(current_airport_data[4])
        twenty_airport_CLC_data[airport_name] = current_airport_data[0]
        print("Processed " + str(airport_count) + " airport datasets")

twenty_airport_CLC_data.to_csv("CLC_Data/Avg_Tables/" + avg_table_title + ".csv")

data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

summary_table = pd.DataFrame(data)

summary_table.to_csv("CLC_Data/Value_Tables/" + value_table_title + '.csv', sep='\t')

"""

#"""
# Average Tables and Value Tables for every 30 year period between 1950-2022 for every airport

#------------------------------------------------------
years = range(1950, 2023)

months = ["September"]

hours = [7, 10, 13, 16]

elevation_def = 1000
#------------------------------------------------------

airport_count = 0
slopes = []
r_val = []
p_val = []
PDO_r_val = []
airports = []
airport_count = 0
era_count = 0

pd.options.mode.chained_assignment = None

def listToString(s):
     
    # initialize an empty string
    str1 = ""
 
    # traverse in the string
    for ele,i in zip(s,range(len(s))):
        if i == len(s)-1:
            str1 += str(ele)
        else:
            str1 += str(ele) + "_"
 
    # return string
    return str1

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# iterate through every 30-year era between 1950 to 2022
for era_beginning in range(1950, 1994):
    era_count += 1
    years = range(era_beginning, era_beginning+30)
    
    avg_table_title = "Airport_CLC_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + "_Months_" + listToString(months) + "_Hours_" + listToString(hours) + "_Elevation_Definition_" + str(elevation_def)

    value_table_title = "Airport_Values_Summary_Table_Years_" + str(years[0]) + "_to_" + str(years[-1]) + "_Months_" + listToString(months) + "_Hours_" + listToString(hours) + "_Elevation_Definition_" + str(elevation_def)

    # clears the dataframe for each era
    twenty_airport_CLC_data = pd.DataFrame({'Year': years})
    slopes = []
    r_val = []
    p_val = []
    PDO_r_val = []
    airports = []
    for file in os.listdir('Airport_Data_Tables'):
        airport_name = file[:file.index(".")]
        elevation = airport_summary["c6"].loc[airport_name]
        if airport_name in airport_acronyms:
            #print(airport_name)
            airport_count += 1
            airports.append(airport_name)
            current_airport_data = calculate_airport_data('Airport_Data_Tables/'+file, years, months, hours, elevation, elevation_def)
            slopes.append(current_airport_data[1])
            r_val.append(current_airport_data[2])
            p_val.append(current_airport_data[3])
            PDO_r_val.append(current_airport_data[4])
            twenty_airport_CLC_data[airport_name] = current_airport_data[0]

    twenty_airport_CLC_data.to_csv("CLC_Data/Avg_Tables/" + avg_table_title + ".csv")

    data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

    summary_table = pd.DataFrame(data)

    summary_table.to_csv("CLC_Data/Value_Tables/" + value_table_title + '.csv', sep='\t')

    loading = "["+ len(range(1950, era_beginning))*"1"+len(range(era_beginning, 1994-1))*"0"+ "]"
    print("Processed " + str(era_count) + " dataset eras")
    print(loading)
#"""


