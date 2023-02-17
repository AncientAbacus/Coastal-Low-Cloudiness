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

# gets airport summary data (c6 = elevation)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

avg_table_title = "Airport_CLC_Summary_Table_" + str(datetime.now().date())

value_table_title = "Airport_Values_Summary_Table_" + str(datetime.now().date())

# function to convert dates and times in given dataset to datetime variables
def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

# function to graph average CLC from 1950-2022 (inclusive)
def graph_airport(airport_data, tf):
    airport_name = airport_data[airport_data.index("/")+1:airport_data.index(".")]

    t = pd.read_csv(airport_data, sep = "\t")

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    timeframe = tf

    # create list for mean CLC % to add to for each year
    CLC = []
    
    # loop from year 1950 to year 2022
    for year in range(1950, 2023):
        CLC_percent_by_month = []

        new_t = t[t.index.year == year]

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

        # add mean of valid monthly CLCs to CLC list
        if timeframe == "Summer":
            CLC.append(np.mean(CLC_percent_by_month))
        if timeframe == "May":
            CLC.append(CLC_percent_by_month[0])
        if timeframe == "June":
            CLC.append(CLC_percent_by_month[1])
        if timeframe == "July":
            CLC.append(CLC_percent_by_month[2])
        if timeframe == "August":
            CLC.append(CLC_percent_by_month[3])
        if timeframe == "September":
            CLC.append(CLC_percent_by_month[4])

    # record mean CLC of airport ignoring missing summers
    mean_missing = np.nanmean(CLC)
    CLC_with_missing = [mean_missing if math.isnan(x) else x for x in CLC]

    PDO = pd.read_csv('PDO_Data.txt',delim_whitespace=True,header=1)
    PDO = PDO.loc[PDO['Year'] >= 1950]
    PDO = PDO.loc[PDO['Year'] <= 2022]
    PDO = PDO[["Year","May","Jun","Jul","Aug","Sep"]]
    PDO['summer_PDO'] = PDO[["May","Jun","Jul","Aug","Sep"]].mean(axis=1)
    PDO['may'] = PDO[["May"]].mean(axis=1)
    PDO['june'] = PDO[["Jun"]].mean(axis=1)
    PDO['july'] = PDO[["Jul"]].mean(axis=1)
    PDO['august'] = PDO[["Aug"]].mean(axis=1)
    PDO['sept'] = PDO[["Sep"]].mean(axis=1)


    x = range(1950,2023)
    y = CLC_with_missing
    if timeframe == "Summer":
        z = PDO['summer_PDO']
    if timeframe == "May":
        z = PDO['may']
    if timeframe == "June":
        z = PDO['june']
    if timeframe == "July":
        z = PDO['july']
    if timeframe == "August":
        z = PDO['august']
    if timeframe == "September":
        z = PDO['sept']

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
    plt.title(airport_data[airport_data.index("/")+1:airport_data.index(".")] + " " + labels[airport_data[airport_data.index("/")+1:airport_data.index(".")]] + " " + timeframe)
    plt.axhline(mean_missing, color='r') # Horizontal line representing average
    
    if timeframe == "Summer":
        plt.ylabel("CLC (% May to Sep. Daytime frequency < 1000m base)")
    if timeframe == "May":
        plt.ylabel("CLC (% May Daytime frequency < 1000m base)")
    if timeframe == "June":
        plt.ylabel("CLC (% June Daytime frequency < 1000m base)")
    if timeframe == "July":
        plt.ylabel("CLC (% July Daytime frequency < 1000m base)")
    if timeframe == "August":
        plt.ylabel("CLC (% August Daytime frequency < 1000m base)")
    if timeframe == "September":
        plt.ylabel("CLC (% Sept Daytime frequency < 1000m base)")

    a, b = np.polyfit(clean_x, clean_y, 1)
    plt.plot(clean_x, a*np.array(clean_x)+b, color="green")

    plt.xlabel("1950:2022\nslope:" + str(round(slope,4)) + " p-value:" + str(round(p_val,4)) + " r-value:" + str(round(r_val,4)))

    if timeframe == "Summer":
        plt.legend(["Summer CLC", "Missing Summer CLC", "Average Summer CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    if timeframe == "May":
        plt.legend(["May CLC", "Missing May CLC", "Average May CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    if timeframe == "June":
        plt.legend(["June CLC", "Missing June CLC", "Average June CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    if timeframe == "July":
        plt.legend(["July CLC", "Missing July CLC", "Average July CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    if timeframe == "August":
        plt.legend(["August CLC", "Missing Aug CLC", "Average August CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))
    if timeframe == "September":
        plt.legend(["September CLC", "Missing Sept CLC", "Average September CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')

    plt.savefig("Airport_Trends/Graphs/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+"_"+timeframe+"_Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.twinx().plot(x, z, color='orange', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.gca().invert_yaxis()

    plt.legend(["PDO"], title="PDO r-value: " + str(round(PDO_r_val,4)), loc="upper left", bbox_to_anchor=(1.05,0.4))

    plt.savefig("Airport_Trends/PDO_Graphs/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+"_"+timeframe+"_PDO_Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.close()

    return CLC, slope, r_val, p_val, PDO_r_val, 

pd.options.mode.chained_assignment = None

check = 0
twenty_airport_CLC_data = pd.DataFrame({'Year': range(1950,2023)})
if not os.path.exists("Airport_Trends/PDO_Graphs/"):
    os.makedirs("Airport_Trends/PDO_Graphs/")
if not os.path.exists("Airport_Trends/Graphs/"):
    os.makedirs("Airport_Trends/Graphs/")

slopes = []
r_val = []
p_val = []
PDO_r_val = []
airports = []
airport_count = 0
"""
for file in os.listdir('Airport_Data_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in airport_acronyms:
        print(airport_name)
        airport_count += 1
        airports.append(airport_name)
        current_airport_data = graph_airport('Airport_Data_Tables/'+file)
        slopes.append(current_airport_data[1])
        r_val.append(current_airport_data[2])
        p_val.append(current_airport_data[3])
        PDO_r_val.append(current_airport_data[4])
        twenty_airport_CLC_data[airport_name] = current_airport_data[0]
        print("Graphed " + str(airport_count) + " airport datasets")

twenty_airport_CLC_data.to_csv(avg_table_title + ".csv")

data = {'airports':airports, 'slopes': slopes, 'r_val':r_val, 'p_val':p_val, 'PDO_r_val':PDO_r_val}  

summary_table = pd.DataFrame(data)

summary_table.to_csv(value_table_title + '.csv', sep='\t')
"""

for file in os.listdir('Airport_Data_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in airport_acronyms:
        print(airport_name)
        airport_count += 1
        seasons = ["Summer", "May", "June", "July", "August", "September"]
        for season in seasons:
            current_airport_data = graph_airport('Airport_Data_Tables/'+file, season)
            print("Graphed " + str(airport_count) + " airport datasets for " + season)

print("Done")



