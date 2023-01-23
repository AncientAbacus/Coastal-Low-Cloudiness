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

labels = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

def graph_airport(airport_data):
    t = pd.read_csv(airport_data, delim_whitespace=True,  names=["year","month","day","hour","cld.frac","cld.base.ft"], parse_dates = {"date" : ["year","month","day","hour",]})

    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y %m %d %H')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    t = t.tz_localize('UTC')
     
    t = t.tz_localize(None)

    t.index = t.index - timedelta(hours=8)

    t.sort_values(by = ['date'])

    t.drop_duplicates

    MIN = 0
    MAX = 1
    
    valid_summer = t[(t.index.month >= 5) & (t.index.month <= 9)]

    hour_dict = {}
    hour_dict_2 = {}

    for i in range(0,24):
        hour_dict_2[i] = []

    year = np.arange(1950, 2023)

    for j in year:
        for i in range(0,24):
            hour_dict[i] = []
        for i in valid_summer[(valid_summer.index.year == j)].index.hour.values:
            hour_dict[i].append(1)
        for i in hour_dict:
            hour_dict_2[i].append(len(hour_dict[i])/153)

    hour = np.arange(24)

    valid_percent_df = pd.DataFrame(data=hour_dict_2, index = year)

    valid_percent_df = valid_percent_df.T

    valid_percent_df = valid_percent_df.iloc[::-1]

    sns.heatmap(valid_percent_df, vmin=MIN, vmax=MAX)

    plt.title(airport_data[airport_data.index("/")+1:airport_data.index(".")] + " " + labels[airport_data[airport_data.index("/")+1:airport_data.index(".")]])

    plt.xlabel("1950:2022")

    plt.ylabel("Hours")

    #plt.show()

    plt.savefig("Heat_Maps/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+" Graph.pdf",  dpi=300, format='pdf', bbox_inches='tight')

    plt.clf()

    return

#graph_airport('Airport_Data_from_Sam/KVNY.cld+hgt.hourly.txt')
#graph_airport('Merged_Files/KVNY.cld+hgt.hourly.txt')

pd.options.mode.chained_assignment = None
#'''
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Older_Files_Tables'):
    airport_name =file[:file.index(".")]
    if airport_name in labels:
        graph_airport('Older_Files_Tables/' + file)
#'''


