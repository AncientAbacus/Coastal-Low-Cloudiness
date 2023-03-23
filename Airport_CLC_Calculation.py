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
import warnings
from Valid import *

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# function to convert dates and times in given dataset to datetime variables
def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

# function to graph average CLC from 1950-2022 (inclusive)
# years - range
# months - list
# hours - list

def calculate_airport_data(airport_data, years, months, hours, elevation, elevation_def):

    airport_name = airport_data[airport_data.index("/")+1:airport_data.index(".")]

    t = pd.read_csv(airport_data, sep = "\t")

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    # create list for mean CLC % to add to for each year
    CLC = []

    month_number_dict = {"January": 1, "Feburary": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}

    # loop through year range
    for year in years:
        CLC_percent_by_month = []

        new_t = t[t.index.year == year]

        valid_frame = new_t[(new_t["cld.base.ft"] != -999) & (new_t["cld.frac"] != -999.9)]
        valid_frame = valid_frame[(((valid_frame["cld.frac"] >= 0.75) & (valid_frame["cld.base.ft"] != 72200)) | (valid_frame["cld.frac"] < 0.75))]

        if check_valid(new_t, year, months, hours):
            for month in months:
                low_observation = valid_frame[(valid_frame["cld.frac"] >= 0.75) & (((valid_frame["cld.base.ft"]*0.3048)+elevation)<elevation_def)]
                low_observation = low_observation[low_observation.index.month == month_number_dict[month]]
                low_observation = low_observation[low_observation.index.hour.isin(hours)]

                valid_observation = valid_frame[valid_frame.index.month == month_number_dict[month]]
                valid_observation = valid_observation[valid_observation.index.hour.isin(hours)]
                
                Avg_CLC = 100*low_observation[low_observation.columns[0]].count()/valid_observation[valid_observation.columns[0]].count()
                CLC_percent_by_month.append(Avg_CLC)
        else:
            CLC_percent_by_month.append(np.nan)

        CLC.append(np.nanmean(CLC_percent_by_month))

    # record mean CLC of airport ignoring missing summers
    mean_missing = np.nanmean(CLC)
    CLC_with_missing = [mean_missing if math.isnan(x) else x for x in CLC]

    PDO = pd.read_csv('PDO_Data.txt',delim_whitespace=True,header=1)
    PDO = PDO.loc[PDO['Year'] >= years[0]]
    PDO = PDO.loc[PDO['Year'] <= years[-1]]
    PDO = PDO[["Year","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]
    PDO['January'] = PDO[["Jan"]].mean(axis=1)
    PDO['Feburary'] = PDO[["Feb"]].mean(axis=1)
    PDO['March'] = PDO[["Mar"]].mean(axis=1)
    PDO['April'] = PDO[["Apr"]].mean(axis=1)
    PDO['May'] = PDO[["May"]].mean(axis=1)
    PDO['June'] = PDO[["Jun"]].mean(axis=1)
    PDO['July'] = PDO[["Jul"]].mean(axis=1)
    PDO['August'] = PDO[["Aug"]].mean(axis=1)
    PDO['September'] = PDO[["Sep"]].mean(axis=1)
    PDO['October'] = PDO[["Oct"]].mean(axis=1)
    PDO['November'] = PDO[["Nov"]].mean(axis=1)
    PDO['December'] = PDO[["Dec"]].mean(axis=1)

    x = years
    y = CLC_with_missing
    included_cols = []
    for month in months:
        included_cols.append(month[:3])
    z = PDO[included_cols].mean(axis=1)

    # record years, summer CLC, and PDO values from non-missing summers
    clean_x = []
    clean_y = []
    clean_z = []
    for year, point, PDO in zip(x, y, z):
        if point != mean_missing:
            clean_x.append(year)
            clean_y.append(point)
            clean_z.append(PDO)

    # calculate p-value of non-missing summers
    # slope is index 0, r is index 2, p is index 3
    slope = scipy.stats.linregress(clean_x, clean_y)[0]
    r_val = scipy.stats.linregress(clean_x, clean_y)[2]
    p_val = scipy.stats.linregress(clean_x, clean_y)[3]
    PDO_r_val = scipy.stats.linregress(clean_y, clean_z)[2]

    return CLC, slope, r_val, p_val, PDO_r_val
