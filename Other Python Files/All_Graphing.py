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
from valid import *

labels = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

def graph_airport(airport_data):

    t = pd.read_csv(airport_data, sep = "\t")

    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    CLC = []

    for year in range(1950,2023):
        new_t = t[t.index.year == year]

        all_s = check_valid(new_t)

        low_s = new_t[(new_t.index.month >= 5) & (new_t.index.month <= 9) & (new_t["cld.base.ft"]<=3280.84) & (new_t["cld.frac"] >= 0.75)]
        low_s = low_s[(low_s.index.hour == 7) | (low_s.index.hour == 10) | (low_s.index.hour == 13) | (low_s.index.hour == 16)]
        

        if all_s[1].size == 0:
            new_CLC = np.nan
        else:
            new_CLC = low_s.size/all_s[1].size

        if not (all_s[0]):
            new_CLC = np.nan

        CLC.append(new_CLC)

    mean_missing = np.nanmean(CLC)*100
    CLC_new = [mean_missing if math.isnan(x) else x*100 for x in CLC]
    x = range(1950,2023)
    y = CLC_new

    fig, ax = plt.subplots()

    graph_df = pd.DataFrame((list(zip(x, y))), columns=["Years", "CLC"])
    graph_df = graph_df.reset_index()

    clean_x = []
    clean_y = []

    for year, point in zip(x, y):
        if point != mean_missing:
            clean_x.append(year)
            clean_y.append(point)

    plt.plot(x, y, color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    avg_graph_df = pd.DataFrame(x, columns=["Years"])
    avg_graph_df["Avg_CLC"] = np.nan

    for index, row in graph_df.iterrows():
        if row["CLC"] == mean_missing:
            avg_graph_df["Avg_CLC"][index] = mean_missing
    
    plt.plot(avg_graph_df["Years"], avg_graph_df["Avg_CLC"], color='red', marker='o', fillstyle='none', linestyle=' ', markersize=5)

    plt.gcf().autofmt_xdate()
    plt.title(airport_data[airport_data.index("/")+1:airport_data.index(".")] + " " + labels[airport_data[airport_data.index("/")+1:airport_data.index(".")]])
    plt.axhline(mean_missing, color='r') # Horizontal line representing average
    
    plt.ylabel("CLC (% May to Sep. Daytime frequency < 1000m base)")

    a, b = np.polyfit(clean_x, clean_y, 1)
    plt.plot(clean_x, a*np.array(clean_x)+b)

    # CALCULATE P-VALUE
    # slope is index 0, r is index 2, p is index 3

    slope = scipy.stats.linregress(clean_x, clean_y)[0]
    r_val = scipy.stats.linregress(clean_x, clean_y)[2]
    p_val = scipy.stats.linregress(clean_x, clean_y)[3]

    plt.xlabel("1950:2022\nslope:" + str(round(slope,4)) + " p-value:" + str(round(p_val,4)) + " r-value:" + str(round(r_val,4)))

    plt.legend(["Summer CLC", "Missing Summer CLC", "Average Summer CLC", "Line of Best Fit"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

    ax.xaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='major')

    #'''
    if "Merged_Files_Tables" in airport_data:
        plt.savefig("Airport_Graphs/Airport_Graphs_M/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+" Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')
    if "Older_Files_Tables" in airport_data:
        plt.savefig("Airport_Graphs/Airport_Graphs_O/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+" Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')
    if "Airport_Data_from_Sam_Tables" in airport_data:
        plt.savefig("Airport_Graphs/Airport_Graphs_S/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+" Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')
    #'''

    #plt.show()

    plt.clf()

    plt.close()

    return

graph_airport('Merged_Files_Tables/KACV.csv')

pd.options.mode.chained_assignment = None

'''
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Merged_Files_Tables'):
    if file[:file.index(".")] in labels:
        graph_airport('Merged_Files_Tables/' + file)
    else:
        print(file[:file.index(".")])
'''

#'''
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Older_Files_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in labels:
        graph_airport('Older_Files_Tables/' + file)
#'''

'''
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Airport_Data_from_Sam_Tables'):
    if file[:file.index(".")] in labels:
        graph_airport('Airport_Data_from_Sam_Tables/' + file)
'''


