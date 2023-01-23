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

twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "PSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

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
    
    for year in range(1950, 2013):
        new_t = t[t.index.year == year]

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan

        elevation = airport_summary["c6"].loc[airport_name]

        all_s = check_valid(new_t)
        # 2:May, 3:June, 4:July, 5:August, 6:Sept

        low_s = new_t[(new_t.index.month >= 5) & (new_t.index.month <= 9) & (new_t["cld.frac"] >= 0.75) & (((new_t["cld.base.ft"]*0.3048)+elevation)<1000)]
        low_s = low_s[(low_s.index.hour == 7) | (low_s.index.hour == 10) | (low_s.index.hour == 13) | (low_s.index.hour == 16)]
        low_s = low_s[(low_s["cld.base.ft"] != -999) & (low_s["cld.frac"] != -999.9)]

        low_s_may = low_s[low_s.index.month == 5]
        low_s_june = low_s[low_s.index.month == 6]
        low_s_july = low_s[low_s.index.month == 7]
        low_s_august = low_s[low_s.index.month == 8]
        low_s_sept = low_s[low_s.index.month == 9]

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

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan

        """
        if year == 1973:
            print(all_s[1].to_string())
            print(all_s[2] and all_s[3] and all_s[4] and all_s[5] and all_s[6])
        """
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

        if (all_s[2] and all_s[3] and all_s[4] and all_s[5] and all_s[6]):
            CLC_percent_by_month.append(may)
            CLC_percent_by_month.append(june)
            CLC_percent_by_month.append(july)
            CLC_percent_by_month.append(august)
            CLC_percent_by_month.append(september)
        else:
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)
            CLC_percent_by_month.append(np.nan)

    return CLC_percent_by_month

fig, ax = plt.subplots()

pd.options.mode.chained_assignment = None

CLC_percent_by_month_dict = {}
check = 0
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Older_Files_TABLES'):
    airport_name = file[:file.index(".")]
    #if airport_name == 'PANT':
    if airport_name in twenty_airport_acronyms:
        check += 1
        print(check)
        avg_month_CLCs = graph_airport('Older_Files_TABLES/'+file)
        CLC_percent_by_month_dict[str(airport_name)] = avg_month_CLCs

CLC_months = pd.DataFrame(data = CLC_percent_by_month_dict)

month_index = []
for year in range(1950,2013):
    month_index.append("May_" + str(year))
    month_index.append("June_" + str(year))
    month_index.append("July_" + str(year))
    month_index.append("August_" + str(year))
    month_index.append("September_" + str(year))

CLC_months = CLC_months.reindex(sorted(CLC_months.columns), axis=1)

CLC_months['average'] = CLC_months.mean(axis=1)

CLC_months.index = month_index

CLC_months.to_csv('CLC_months.csv', sep='\t')

ex_t = pd.read_csv('MonLCfl_monthlyLowCloudFreq_AP20_7101316PST_wNA_090413_txt_111022.txt', sep = " ", header = 0, index_col = 0)

ex_t['average'] = ex_t.mean(axis=1)

ex_t = ex_t.reindex(sorted(ex_t.columns), axis=1)

ex_t.to_csv('Rachel_months.csv', sep='\t')

ex_x = range(1950,2013)

ex_t_by_year = ex_t.groupby(np.arange(len(ex_t))//5).mean()

CLC_months_by_year = CLC_months.groupby(np.arange(len(CLC_months))//5).mean()

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')

difference = pd.DataFrame(round(ex_t_by_year['average']-CLC_months_by_year['average'], 6))
print(difference.to_string())

plt.plot(ex_x, ex_t_by_year['average'], color='green', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.plot(ex_x, CLC_months_by_year['average'], color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.title("Pacific Rim Avg vs EX Pacific Rim Avg")

plt.savefig("Versus Pacfic Rim Graph 12-5.pdf", dpi=300, format='pdf', bbox_inches='tight')

plt.show()





