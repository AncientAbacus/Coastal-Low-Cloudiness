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

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

# gets airport summary data (c4 = latitude)
airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

def date_convert(date_to_convert):
    return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

bar_plot_data = pd.read_csv('Twenty_Two_Master_Table_1-26.csv', sep = "\t", usecols=range(1,6), index_col = 'airports')

airports = bar_plot_data.index

bar_plot_data['latitude'] = range(0,22)

for airport in airports:
    bar_plot_data['latitude'].loc[airport] = airport_summary['c4'].loc[airport]

bar_plot_data = bar_plot_data.sort_values(by=['latitude'], ascending = True)

print(bar_plot_data)

airports = bar_plot_data.index

slopes = bar_plot_data[bar_plot_data.columns[0]]

colors = []

for airport in airports:
    if bar_plot_data['p_val'].loc[airport] < 0.05:
        colors.append('red')
    else:
        colors.append('blue')



plt.barh(airports, slopes, color=colors)

colors = {'p-value < 0.05':'red', 'p-value >= 0.05':'blue'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.xlabel("Slope")
plt.ylabel("Airports")
plt.title("Coastal Airport Slopes")
plt.savefig('Twenty_Two_Bar_Plot_1-27-23.pdf', bbox_inches='tight')

print(bar_plot_data)


