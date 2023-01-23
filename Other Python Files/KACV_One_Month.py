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

labels = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

year_to_check = 1973
airport_to_check = "PANT"

def graph_airport(airport_data):

    airport_name = airport_data[airport_data.index("/")+1:airport_data.index(".")]

    print(airport_name)

    t = pd.read_csv(airport_data, sep = "\t")

    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    CLC = []

    CLC_percent_by_month = []

    may = np.nan
    june = np.nan
    july = np.nan
    august = np.nan
    september = np.nan

    elevation = airport_summary["c6"].loc[airport_name]
    
    for year in range(year_to_check, year_to_check+1):
        year_t = t[t.index.year == year]

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan

        all_s = check_valid(year_t)

        # 2:May, 3:June, 4:July, 5:August, 6:Sept

        low_s = year_t[(year_t.index.month >= 5) & (year_t.index.month <= 9) & (year_t["cld.frac"] >= 0.75) & (((year_t["cld.base.ft"]*0.3048)+elevation)<1000)]
        low_s = low_s[(low_s.index.hour == 7) | (low_s.index.hour == 10) | (low_s.index.hour == 13) | (low_s.index.hour == 16)]

        low_s_may = low_s[low_s.index.month == 5]
        low_s_june = low_s[low_s.index.month == 6]
        low_s_july = low_s[low_s.index.month == 7]
        low_s_august = low_s[low_s.index.month == 8]
        low_s_sept = low_s[low_s.index.month == 9]

        year_t = year_t[(year_t.index.month >= 5) & (year_t.index.month <= 9)] 
        year_t = year_t[(year_t["cld.base.ft"] != -999) & (year_t["cld.frac"] != -999.9)]
        year_t = year_t[(((year_t["cld.frac"] > 0.7) & (year_t["cld.base.ft"] != 72200)) | ((year_t["cld.frac"] < 0.7) & (year_t["cld.base.ft"] <= 72200)))]
        year_t = year_t[(year_t.index.hour == 7) | (year_t.index.hour == 10) | (year_t.index.hour == 13) | (year_t.index.hour == 16)]
        all_s_may = year_t[(year_t.index.month == 5)]
        all_s_june = year_t[year_t.index.month == 6]
        all_s_july = year_t[year_t.index.month == 7]
        all_s_august = year_t[year_t.index.month == 8]
        all_s_sept = year_t[year_t.index.month == 9]

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan


        if all_s[2]:
        	print("may numerator: " + str(low_s_may[low_s_may.columns[0]].count()))
        	print("may denominator: " + str(all_s_may[all_s_may.columns[0]].count()))
        	may = (100*low_s_may[low_s_may.columns[0]].count())/all_s_may[all_s_may.columns[0]].count()
        if all_s[3]:
        	print("june numerator: " + str(low_s_june[low_s_june.columns[0]].count()))
        	print("june denominator: " + str(all_s_june[all_s_june.columns[0]].count()))
        	june = (100*low_s_june[low_s_june.columns[0]].count())/all_s_june[all_s_june.columns[0]].count()
        if all_s[4]:
        	print("july numerator: " + str(low_s_july[low_s_july.columns[0]].count()))
        	print("july denominator: " + str(all_s_july[all_s_july.columns[0]].count()))
        	july = (100*low_s_july[low_s_july.columns[0]].count())/all_s_july[all_s_july.columns[0]].count()
        print(all_s[5])
        if all_s[5]:
            print("august numerator: " + str(low_s_august[low_s_august.columns[0]].count()))
            print("august denominator: " + str(all_s_august[all_s_august.columns[0]].count()))
            august = (100*low_s_august[low_s_august.columns[0]].count())/all_s_august[all_s_august.columns[0]].count()
        if all_s[6]:
        	print("sept numerator: " + str(low_s_sept[low_s_sept.columns[0]].count()))
        	print("sept denominator: " + str(all_s_sept[all_s_sept.columns[0]].count()))
        	september = (100*low_s_sept[low_s_sept.columns[0]].count())/all_s_sept[all_s_sept.columns[0]].count()

        CLC_percent_by_month.append(may)
        CLC_percent_by_month.append(june)
        CLC_percent_by_month.append(july)
        CLC_percent_by_month.append(august)
        CLC_percent_by_month.append(september)

    return CLC_percent_by_month

pd.options.mode.chained_assignment = None

CLC_percent_by_month_dict = {}
airport_file = 'Twenty_Tables/' + airport_to_check + '.csv'
avg_month_CLCs = graph_airport(airport_file)
CLC_percent_by_month_dict[str(airport_file[airport_file.index("/")+1:airport_file.index(".")])] = avg_month_CLCs

CLC_months = pd.DataFrame(data = CLC_percent_by_month_dict)

month_index = []
for year in range(year_to_check, year_to_check+1):
    month_index.append("May_" + str(year))
    month_index.append("June_" + str(year))
    month_index.append("July_" + str(year))
    month_index.append("August_" + str(year))
    month_index.append("September_" + str(year))

CLC_months['average'] = CLC_months.mean(axis=1)

CLC_months = CLC_months.reindex(sorted(CLC_months.columns), axis=1)

CLC_months.index = month_index

print("Gino's")
print(CLC_months[airport_to_check])

ex_t = pd.read_csv('MonLCfl_monthlyLowCloudFreq_AP20_7101316PST_wNA_090413_txt_111022.txt', sep = " ", header = 0, index_col = 0)
print(ex_t)

ex_t['average'] = ex_t.mean(axis=1)

ex_t = ex_t.reindex(sorted(ex_t.columns), axis=1)

ex_t = ex_t[airport_to_check].to_string()
print(ex_t)

#print(ex_t[ex_t.index == 'Sep_' + str(year_to_check)])

print("Rachel's")
#print(ex_t[(ex_t.index == 'Sep_' + str(year_to_check)) | (ex_t.index == 'Aug_' + str(year_to_check)) | (ex_t.index == 'Jun_' + str(year_to_check)) | (ex_t.index == 'Jul_' + str(year_to_check)) | (ex_t.index == 'May_' + str(year_to_check))])

#ex_t_by_year = ex_t.groupby(np.arange(len(ex_t))//5).mean()

CLC_months_by_year = CLC_months.groupby(np.arange(len(CLC_months))//5).mean()





