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

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

# twenty two aiport acronyms used for pacific rim summary
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

# function to get monthly summer average CLC from 1950-2022 (inclusive) for all twenty airports
def graph_airport(airport_data):

    airport_name = airport_data[airport_data.index("/")+1:airport_data.index(".")]

    print(airport_name)

    t = pd.read_csv(airport_data, sep = "\t")

    # function to convert dates and times in given dataset to datetime variables
    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    # create list for mean CLC % to add to for each year
    CLC = []

    CLC_percent_by_month = []

    may = np.nan
    june = np.nan
    july = np.nan
    august = np.nan
    september = np.nan
    
    # loop from year 1950 to year 2022
    for year in range(1950, 2023):
        new_t = t[t.index.year == year]

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan

        # get elevation of current airport
        elevation = airport_summary["c6"].loc[airport_name]

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

        may = np.nan
        june = np.nan
        july = np.nan
        august = np.nan
        september = np.nan

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

    #return avg CLC for every summer month from 1950 to 2022
    return CLC_percent_by_month

# get PDO data from summer months from 1950 to 2022 (inclusidve)
PDO = pd.read_csv('PDO_Data.txt',delim_whitespace=True,header=1)
PDO = PDO.loc[PDO['Year'] >= 1950]
PDO = PDO.loc[PDO['Year'] <= 2022]
PDO = PDO[["Year","May","Jun","Jul","Aug","Sep"]]
PDO['summer_PDO'] = PDO[["May","Jun","Jul","Aug","Sep"]].mean(axis=1)

fig, ax = plt.subplots()

pd.options.mode.chained_assignment = None

# record avg summer CLC % for each airport for each year 
CLC_percent_by_month_dict = {}
check = 0
for file in os.listdir('Twenty_Tables'):
    airport_name = file[:file.index(".")]
    check += 1
    print("Graphing " + str(check) + " airport")
    avg_month_CLCs = graph_airport('Twenty_Tables/'+file)
    CLC_percent_by_month_dict[str(airport_name)] = avg_month_CLCs

# label index by summer month and years from 1950-2022
CLC_months = pd.DataFrame(data = CLC_percent_by_month_dict)

month_index = []
for year in range(1950,2023):
    month_index.append("May_" + str(year))
    month_index.append("June_" + str(year))
    month_index.append("July_" + str(year))
    month_index.append("August_" + str(year))
    month_index.append("September_" + str(year))

CLC_months = CLC_months.reindex(sorted(CLC_months.columns), axis=1)

CLC_months['average'] = CLC_months.mean(axis=1)

CLC_months.index = month_index

CLC_months.to_csv('2022_CLC_months.csv', sep='\t')

# 2012 Pacific Rim summary
ex_t = pd.read_csv('MonLCfl_monthlyLowCloudFreq_AP20_7101316PST_wNA_090413_txt_111022.txt', sep = " ", header = 0, index_col = 0)

ex_t['average'] = ex_t.mean(axis=1)

ex_t = ex_t.reindex(sorted(ex_t.columns), axis=1)

ex_t.to_csv('2022_Rachel_months.csv', sep='\t')

ex_x = range(1950,2013)

x = range(1950,2023)

ex_t_by_year = ex_t.groupby(np.arange(len(ex_t))//5).mean()

CLC_months_by_year = CLC_months.groupby(np.arange(len(CLC_months))//5).mean()

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')

difference = pd.DataFrame(round(ex_t_by_year['average']-CLC_months_by_year['average'], 6))
print(difference.to_string())

# graph 2012 and 2022 data for comparison
plt.plot(x, CLC_months_by_year['average'], color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.plot(ex_x, ex_t_by_year['average'], color='green', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.xlabel("1950:2022")

plt.ylabel("Average Summer CLC %")

plt.legend(["Up to 2022\n(Updated)", "Up to 2012"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

plt.title("Pacific Rim CLC Trend (20 Airport Averages)")

plt.savefig("2022 Versus Pacfic Rim Graph 1-5.pdf", dpi=300, format='pdf', bbox_inches='tight')

# graph 2012 and 2022 data for comparison along with PDO values
PDO_r_val = scipy.stats.linregress(x, CLC_months_by_year['average'])[2]

z = PDO["summer_PDO"]

plt.twinx().plot(x, z, color='orange', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.gca().invert_yaxis()

plt.legend(["PDO"], title="PDO r-value: " + str(round(PDO_r_val,4)), loc="upper left", bbox_to_anchor=(1.05,0.4))

plt.savefig("PDO 2022 Versus Pacfic Rim Graph 1-5.pdf", dpi=300, format='pdf', bbox_inches='tight')

plt.show()





