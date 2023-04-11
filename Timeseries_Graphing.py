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

#"""
# Years: 1950-2022, Months: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 1000m

years = range(1950, 2023)
months = ["May", "June", "July", "August", "September"]
hours = [7, 10, 13, 16]
elevation_def = 1000

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

airport_avg_data = pd.read_csv("CLC_Data/Avg_Tables/Airport_CLC_Summary_Table_Years_1950_to_2022_Months_May_June_July_August_September_Hours_7_10_13_16.csv")
airport_value_data = pd.read_csv("CLC_Data/Value_Tables/Airport_Values_Summary_Table_Years_1950_to_2022_Months_May_June_July_August_September_Hours_7_10_13_16.csv")

for airport in airport_acronyms:
    CLC = airport_avg_data[airport]
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

    fig, ax = plt.subplots()

    graph_df = pd.DataFrame((list(zip(x, y))), columns=["Years", "CLC"])
    graph_df = graph_df.reset_index()

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

    # plot nonmissing summers in blue
    plt.plot(x, y, color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    avg_graph_df = pd.DataFrame(x, columns=["Years"])
    avg_graph_df["Avg_CLC"] = np.nan

    # plot missing summers in red
    for index, row in graph_df.iterrows():
        if row["CLC"] == mean_missing:
            avg_graph_df["Avg_CLC"][index] = mean_missing

    plt.plot(avg_graph_df["Years"], avg_graph_df["Avg_CLC"], color='red', marker='o', fillstyle='none', linestyle=' ', markersize=5)

    plt.gcf().autofmt_xdate()
    month_names = ""
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

    details = airport + "_Years_" + str(years[0]) + "_to_" + str(years[-1]) + "_Months_" + listToString(months) + "_Hours_" + listToString(hours) + "_Elevation_Definition_" + str(elevation_def)

    plt.title(labels[airport] + details)
    plt.axhline(mean_missing, color='r') # Horizontal line representing average

    if len(months) > 1:
        plt.ylabel("CLC (% " + months[0][:3] + ". to " + months[-1][:3] + ". Daytime frequency < 1000m base)", fontsize=8)
    else:
        plt.ylabel("CLC (% " + months[0][:3] + " Daytime frequency < 1000m base)")

    a, b = np.polyfit(clean_x, clean_y, 1)
    plt.plot(clean_x, a*np.array(clean_x)+b, color="green")

    plt.xlabel("1950:2022\nslope:" + str(round(slope,4)) + " p-value:" + str(round(p_val,4)) + " r-value:" + str(round(r_val,4)))

    if len(months) > 1:
        plt.legend([months[0][:3] + ". to " + months[-1][:3] + ". Summer CLC", \
            "Missing " + months[0][:3] + ". to " + months[-1][:3] + ". Summer CLC", \
            "Average " + months[0][:3] + ". to " + months[-1][:3] + ". Summer CLC", \
            "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    else:
        plt.legend([months[0][:3] + ". Summer CLC", "Missing " + months[0][:3] + ". Summer CLC", \
            "Average " + months[0][:3] + ". Summer CLC", "Line of Best Fit"], \
            title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')

    plt.savefig("Airport_Trends/Timeseries_Graphs/" + details + "_Timeseries.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.twinx().plot(x, z, color='orange', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.gca().invert_yaxis()

    plt.legend(["PDO"], title="PDO r-value: " + str(round(PDO_r_val,4)), loc="upper left", bbox_to_anchor=(1.05,0.4))

    plt.savefig("Airport_Trends/PDO_Timeseries_Graphs/" + details + "PDO_Timeseries.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.close()
#"""
