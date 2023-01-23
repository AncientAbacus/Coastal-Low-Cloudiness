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

def get_props(year_frame):
    all_s = year_frame[(year_frame.index.month >= 5) & (year_frame.index.month <= 9)]
    """
    all_s = all_s[(all_s["cld.base.ft"] != -999) & (all_s["cld.frac"] != -999.9)]
    all_s = all_s[(((all_s["cld.frac"] >= 0.75) & (all_s["cld.base.ft"] != 72200)) | ((all_s["cld.frac"] < 0.75) & (all_s["cld.base.ft"] <= 72200)))]
    """

    all_s_may = all_s[all_s.index.month == 5]
    all_s_june = all_s[all_s.index.month == 6]
    all_s_july = all_s[all_s.index.month == 7]
    all_s_august = all_s[all_s.index.month == 8]
    all_s_sept = all_s[all_s.index.month == 9]

    all_s_may_prop_list = []
    all_s_june_prop_list = []
    all_s_july_prop_list = []
    all_s_august_prop_list = []
    all_s_sept_prop_list = []

    #print(all_s_sept[all_s_sept.index.hour == 1])
    #print(all_s_sept[all_s_sept.index.hour == 1].shape[0])

    for i in range(0,24):
        all_s_may_prop_list.append(all_s_may[all_s_may.index.hour == i].shape[0]/31)

    for i in range(0,24):
        all_s_june_prop_list.append(all_s_june[all_s_june.index.hour == i].shape[0]/30)

    for i in range(0,24):
        all_s_july_prop_list.append(all_s_july[all_s_july.index.hour == i].shape[0]/31)

    for i in range(0,24):
        all_s_august_prop_list.append(all_s_august[all_s_august.index.hour == i].shape[0]/31)

    for i in range(0,24):
        all_s_sept_prop_list.append(all_s_sept[all_s_sept.index.hour == i].shape[0]/30)

    return [all_s_may_prop_list, all_s_june_prop_list, all_s_july_prop_list, all_s_august_prop_list, all_s_sept_prop_list]

yearly_dict = {}
for year in range(1950,2013):
    for month in range(5,10):
        yearly_dict[str(year) + "-" + str(month)] = [[0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24];

def graph_airport(airport_data):

    t = pd.read_csv(airport_data, sep = "\t")

    def date_convert(date_to_convert):
         return datetime.strptime(date_to_convert, '%Y-%m-%d %H:%M:%S')

    t['date'] = t['date'].apply(date_convert)

    t = t.set_index('date')

    for year in range(1950,2013):

        year_t = t[t.index.year == year]

        for month, i in zip(range(5,10), range(0,6)):
            yearly_dict[str(year) + "-" + str(month)] = get_props(year_t)[i]

        #print(get_props(year_t))
        #print(yearly_dict)

    valid_percent_df = pd.DataFrame(data=yearly_dict)

    valid_percent_df = valid_percent_df.iloc[::-1]

    #print(valid_percent_df.to_string())

    sns.heatmap(valid_percent_df, linewidths=0, vmin=0, vmax=1, cmap="coolwarm")

    plt.title(airport_data[airport_data.index("/")+1:airport_data.index(".")] + " " + labels[airport_data[airport_data.index("/")+1:airport_data.index(".")]])

    ticks = np.arange(0, 63*5, 5)
    tick_labels = range(1950,2013)
    plt.xticks(ticks, tick_labels, fontweight='bold', fontsize='4', horizontalalignment='right')
    plt.xlabel("1950:2012")

    plt.ylabel("Hours")

    #plt.show()

    plt.savefig("Heat_Maps_Months/" + airport_data[airport_data.index("/")+1:airport_data.index(".")]+" Graph.pdf", format='pdf')

    plt.clf()

pd.options.mode.chained_assignment = None
#graph_airport('Merged_Files_Tables/KACV.csv')

#'''
for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Older_Files_Tables'):
    airport_name = file[:file.index(".")]
    if airport_name in labels:
        print(airport_name)
        graph_airport('Older_Files_Tables/' + file)
#'''

