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
def graph_airport(airport_data, years, months, hours, elevation):

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
                low_observation = valid_frame[(valid_frame["cld.frac"] >= 0.75) & (((valid_frame["cld.base.ft"]*0.3048)+elevation)<1000)]
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
        for ele in s:
            str1 += str(ele) + "_"
     
        # return string
        return str1

    plt.title(airport_data[airport_data.index("/")+1:airport_data.index(".")] + " " + labels[airport_data[airport_data.index("/")+1:airport_data.index(".")]] + " Years " + str(years[0]) + "_to_" + str(years[-1]) +  listToString(months) + " Hours " + listToString(hours))
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

    if not os.path.exists("Airport_Trends/PDO_Graphs/"):
        os.makedirs("Airport_Trends/PDO_Graphs/")
    if not os.path.exists("Airport_Trends/Graphs/"):
        os.makedirs("Airport_Trends/Graphs/")
    if not os.path.exists("CLC_Data"):
        os.makedirs("CLC_Data")

    plt.savefig("Airport_Trends/Graphs/" + airport_data[airport_data.index("/")+1:airport_data.index(".")] + "_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours) + "_Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.twinx().plot(x, z, color='orange', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.gca().invert_yaxis()

    plt.legend(["PDO"], title="PDO r-value: " + str(round(PDO_r_val,4)), loc="upper left", bbox_to_anchor=(1.05,0.4))

    plt.savefig("Airport_Trends/PDO_Graphs/" + airport_data[airport_data.index("/")+1:airport_data.index(".")] + "_Years_" + str(years[0]) + "_to_" + str(years[-1]) + listToString(months) + "_Hours_" + listToString(hours) + "_PDO_Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.close()

    return CLC, slope, r_val, p_val, PDO_r_val
