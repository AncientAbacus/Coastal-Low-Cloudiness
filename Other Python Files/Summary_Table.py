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

labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

#airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
#"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB"]

airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

table_title = "Airport_Summary_Table_" + str(datetime.now().date())

# gets airport summary data
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',\
    delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "elevation", "c7"])

def graph_airport(airport_data):

    t = pd.read_csv(airport_data, sep = "\t")

    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    CLC = []

    PDO = pd.read_csv('PDO_Data.txt',delim_whitespace=True,header=1)
    PDO = PDO.loc[PDO['Year'] >= 1950]
    PDO = PDO.loc[PDO['Year'] <= 2022]
    PDO = PDO[["Year","May","Jun","Jul","Aug","Sep"]]
    PDO['summer_PDO'] = PDO[["May","Jun","Jul","Aug","Sep"]].mean(axis=1)

    for year in range(1950,2023):
        CLC_percent_by_month = []

        new_t = t[t.index.year == year]

        # get elevation of current airport
        elevation = airport_summary["elevation"].loc[airport_name]

        # check if current year's summer is valid 
        # (>=25% of all possible observations are < 72200 feet and < 0.75 cloud fraction FOR EACH HOUR)
        # indices: 2 = May, 3 = June, 4 = July, 5 = August, 6 = Sept
        all_s = check_valid(new_t)

        # CLC filter
        low_s = new_t[(new_t.index.month >= 5) & (new_t.index.month <= 9) & (new_t["cld.frac"] >= 0.75) & (((new_t["cld.base.ft"]*0.3048)+elevation)<1000)]
        low_s = low_s[(low_s.index.hour == 7) | (low_s.index.hour == 10) | (low_s.index.hour == 13) | (low_s.index.hour == 16)]
        low_s = low_s[(low_s["cld.base.ft"] != -999) & (low_s["cld.frac"] != -999.9)]

        # CLC filter by month
        low_s_may = low_s[low_s.index.month == 5]
        low_s_june = low_s[low_s.index.month == 6]
        low_s_july = low_s[low_s.index.month == 7]
        low_s_august = low_s[low_s.index.month == 8]
        low_s_sept = low_s[low_s.index.month == 9]

        # filter all valid observations for each summer month
        new_t = new_t[(new_t.index.month >= 5) & (new_t.index.month <= 9)] 
        new_t = new_t[(new_t["cld.base.ft"] != -999) & (new_t["cld.frac"] != -999.9)]
        new_t = new_t[(((new_t["cld.frac"] >= 0.75) & (new_t["cld.base.ft"] != 72200)) | ((new_t["cld.frac"] < 0.75) & (new_t["cld.base.ft"] <= 72200)))]
        all_s_may = new_t[new_t.index.month == 5]
        all_s_may = all_s_may[(all_s_may.index.hour == 7) | (all_s_may.index.hour == 10) | (all_s_may.index.hour == 13) | (all_s_may.index.hour == 16)]
        all_s_june = new_t[new_t.index.month == 6]
        all_s_june = all_s_june[(all_s_june.index.hour == 7) | (all_s_june.index.hour == 10) | (all_s_june.index.hour == 13) | (all_s_june.index.hour == 16)]
        all_s_july = new_t[new_t.index.month == 7]
        all_s_july = all_s_july[(all_s_july.index.hour == 7) | (all_s_july.index.hour == 10) | (all_s_july.index.hour == 13) | (all_s_july.index.hour == 16)]
        all_s_august = new_t[new_t.index.month == 8]
        all_s_august = all_s_august[(all_s_august.index.hour == 7) | (all_s_august.index.hour == 10) | (all_s_august.index.hour == 13) | (all_s_august.index.hour == 16)]
        all_s_sept = new_t[new_t.index.month == 9]
        all_s_sept = all_s_sept[(all_s_sept.index.hour == 7) | (all_s_sept.index.hour == 10) | (all_s_sept.index.hour == 13) | (all_s_sept.index.hour == 16)]

        if all_s[2]:
            may = 100*low_s_may[low_s_may.columns[0]].count()/all_s_may[all_s_may.columns[0]].count()
        if all_s[3]:
            june = 100*low_s_june[low_s_june.columns[0]].count()/all_s_june[all_s_june.columns[0]].count()
        if all_s[4]:
            july = 100*low_s_july[low_s_july.columns[0]].count()/all_s_july[all_s_july.columns[0]].count()
        if all_s[5]:
            august = 100*low_s_august[low_s_august.columns[0]].count()/all_s_august[all_s_august.columns[0]].count()
        if all_s[6]:
            september = 100*low_s_sept[low_s_sept.columns[0]].count()/all_s_sept[all_s_sept.columns[0]].count()

        # if all months are valid, add all CLCs to CLC_percent_by_month
        if (all_s[2] and all_s[3] and all_s[4] and all_s[5] and all_s[6]):
            CLC_percent_by_month.append(may)
            CLC_percent_by_month.append(june)
            CLC_percent_by_month.append(july)
            CLC_percent_by_month.append(august)
            CLC_percent_by_month.append(september)
        # else add 5 nan values to CLC_percent_by_month
        else:
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)

        #add mean of valid monthly CLCs to CLC list
        CLC.append(np.mean(CLC_percent_by_month))

    mean_missing = np.nanmean(CLC)
    CLC_new = [mean_missing if math.isnan(x) else x for x in CLC]
    x = range(1950,2023)
    y = CLC_new
    z = PDO["summer_PDO"]

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

    return slope, r_val, p_val, PDO_r_val

slopes = []
r_val = []
p_val = []
PDO_r_val = []
airports = []

pd.options.mode.chained_assignment = None
airport_count = 0
for file in os.listdir('Airport_Data_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in airport_acronyms:
        print(airport_name)
        airport_count += 1
        print(airport_count)
        airports.append(airport_name)
        current_airport_data = graph_airport('Airport_Data_Tables/'+file)
        slopes.append(current_airport_data[0])
        r_val.append(current_airport_data[1])
        p_val.append(current_airport_data[2])
        PDO_r_val.append(current_airport_data[3])
print("Done")

data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

master_table = pd.DataFrame(data)

master_table.to_csv(table_title + '.csv', sep='\t')

