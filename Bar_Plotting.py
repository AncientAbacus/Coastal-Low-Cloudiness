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

"""
month = "May"

pd.options.mode.chained_assignment = None

labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# gets airport summary data
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "latitude", "c5", "elevation", "c7"])

def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

def bar_plot(season):
    bar_plot_title = "Bar_Plot_By_Elevation_" + season + "_" + str(datetime.now().date())

    bar_plot_data = pd.read_csv("Airport_Monthly_Values_Summary_Table_2023-02-22_" + season + ".csv", sep = "\t", usecols=range(1,6), index_col = 'airports')

    airports = bar_plot_data.index

    bar_plot_data['latitude'] = range(0,22)
    for airport in airports:
        bar_plot_data['latitude'].loc[airport] = airport_summary['latitude'].loc[airport]

    #bar_plot_data['elevation'] = range(0,22)
    #for airport in airports:
    #    bar_plot_data['elevation'].loc[airport] = airport_summary['elevation'].loc[airport]

    bar_plot_data = bar_plot_data.sort_values(by=['latitude'], ascending = True)

    airports = bar_plot_data.index

    slopes = bar_plot_data[bar_plot_data.columns[0]]

    colors = []

    for airport in airports:
        if bar_plot_data['p_val'].loc[airport] < 0.05:
            colors.append('red')
        else:
            colors.append('blue')

    plt.rcParams.update({'font.size': 5})
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)

    plt.barh(airports, slopes, color=colors)

    colors = {'p-value < 0.05':'red', 'p-value >= 0.05':'blue'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.xlabel("Slope")
    plt.ylabel("Airports")
    plt.title(bar_plot_title)
    plt.savefig("Airport_Trends/Bar_Plots/" + bar_plot_title + '.pdf', bbox_inches='tight')
    plt.clf()

for season in ["May", "June", "July", "August", "September"]:
    bar_plot(season)

"""

"""
# Years: every 30 year period between 1950-2022 Months: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 1000m

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

airport_sig_slope_counts = {}

for airport in twenty_airport_acronyms:
    # negative slope, positive slope
    airport_sig_slope_counts[airport] = [0, 0]

for file in os.listdir("CLC_Data/Value_Tables"):
    if "Elevation_Definiton_400" not in file and "Airport_Values_Summary_Table" in file and int(file[43:47]) - int(file[35:39]) == 29:
        airport_data = pd.read_csv("CLC_Data/Value_Tables/" + file, sep = "\t", index_col = "airports")
        for airport in twenty_airport_acronyms:
            if airport_data.loc[airport,'p_val'] < 0.05:
                if airport_data.loc[airport,'slopes'] > 0:
                    airport_sig_slope_counts[airport][1] += 1
                elif airport_data.loc[airport,'slopes'] < 0:
                    airport_sig_slope_counts[airport][0] += 1

#print(airport_sig_slope_counts)

fig, ax = plt.subplots(layout='constrained')

names = list(airport_sig_slope_counts.keys())
X_axis = np.arange(len(names))
neg_values = list(airport_sig_slope_counts.values())
neg_values = [e0[0]*-1 for e0 in neg_values]
pos_values = list(airport_sig_slope_counts.values())
pos_values = [e1[1] for e1 in pos_values]

print(pos_values)


plt.bar(X_axis - 0.2, neg_values, 0.4, label = 'Negative_Significant_Slopes')
plt.bar(X_axis + 0.2, pos_values, 0.4, label = 'Positive_Significant_Slopes')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xticks(X_axis, names)
ax.set_ylabel('30 Year Period Count')
ax.set_xlabel('Coastal_Airports')
ax.set_title('Significant decreasing (blue) and increasing (orange) 30-year trends in CLC from a moving window analysis by airport\nMonths: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 1000m')
ax.legend(loc='lower right')
#ax.set_ylim(-10, 10)
#Airport_Trends/Bar_Plots/30-year moving window Months: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 1000m Bar Plot

plt.show()

"""

#"""
# Years: every 30 year period between 1950-2022 Months: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 400m

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

airport_sig_slope_counts = {}

for airport in twenty_airport_acronyms:
    # negative slope, positive slope
    airport_sig_slope_counts[airport] = [0, 0]

for file in os.listdir("CLC_Data/Value_Tables"):
    if "Elevation_Definiton_400" in file and "Airport_Values_Summary_Table" in file and int(file[43:47]) - int(file[35:39]) == 29:
        airport_data = pd.read_csv("CLC_Data/Value_Tables/" + file, sep = "\t", index_col = "airports")
        for airport in twenty_airport_acronyms:
            print(airport_data.loc[airport,'p_val'])
            if airport_data.loc[airport,'p_val'] < 0.05:
                if airport_data.loc[airport,'slopes'] > 0:
                    airport_sig_slope_counts[airport][1] += 1
                elif airport_data.loc[airport,'slopes'] < 0:
                    airport_sig_slope_counts[airport][0] += 1

print(airport_sig_slope_counts)

fig, ax = plt.subplots(layout='constrained')

names = list(airport_sig_slope_counts.keys())
X_axis = np.arange(len(names))
neg_values = list(airport_sig_slope_counts.values())
neg_values = [e0[0]*-1 for e0 in neg_values]
pos_values = list(airport_sig_slope_counts.values())
pos_values = [e1[1] for e1 in pos_values]

print(pos_values)


plt.bar(X_axis - 0.2, neg_values, 0.4, label = 'Negative_Significant_Slopes')
plt.bar(X_axis + 0.2, pos_values, 0.4, label = 'Positive_Significant_Slopes')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xticks(X_axis, names)
ax.set_ylabel('30 Year Period Count')
ax.set_xlabel('Coastal_Airports')
ax.set_title('Significant decreasing (blue) and increasing (orange) 30-year trends in CLC from a moving window analysis by airport\nMonths: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 400m')
ax.legend(loc='lower right')
#ax.set_ylim(-10, 10)

#plt.savefig("Airport_Trends/Bar_Plots/30-year moving window Months: May-Sept., Hours: 7, 10, 13, 16, Elevation Definition: 400m Bar Plot.pdf",  dpi=300, format='pdf', bbox_inches='tight')

plt.show()

#"""






