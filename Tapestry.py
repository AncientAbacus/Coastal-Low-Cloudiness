import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pandas as pd
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
from Airport_CLC_Calculation import *
from matplotlib.patches import Rectangle


# Make an image plot "tapestry" airports by latitude on y-axis, 
# months on x-axis, colors slope, mark significant. (should see nor/so Cal sig June, SoCal sig later summer signal)

#"""
# Month by Latitude graph. Monthly CLC slopes. In red if significant. 1 plot

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

#-------------------------------------------------------
years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

elevation_def = 1000

sort_by = "latitude"

y_axis = airport_acronyms

x_axis = months
#-------------------------------------------------------

def tapestry():

    airport_summary = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "latitude", "c5", "elevation", "c7"])

    details = "Months_" + listToString(months) + "_Hours_" + listToString(hours) + "_Elevation_Definition_" + str(elevation_def) 

    # gets acronyms for all airports
    labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
    labels = dict(labels.values[1::])

    tapestry_data = pd.DataFrame()

    print(pd.read_csv("CLC_Data/Value_Tables/Airport_Values_Summary_Table_Years_1950_to_2022_Months_" + "June" + "_Hours_7_10_13_16_Elevation_Definition_1000.csv", sep = "\t"))

    significants = {}

    for month in months:
        current_data = pd.read_csv("CLC_Data/Value_Tables/Airport_Values_Summary_Table_Years_1950_to_2022_Months_" + month + "_Hours_7_10_13_16_Elevation_Definition_1000.csv", sep = "\t")
        tapestry_data[month] = current_data[["airports", "slopes"]].set_index("airports")
        significant = current_data[current_data['p_val'] < 0.05]
        significants[month] = significant['airports'].to_list()

    print(significants)

    tapestry_data[sort_by] = range(0,22)

    for airport in airport_acronyms:
        tapestry_data[sort_by].loc[airport] = airport_summary[sort_by].loc[airport]

    tapestry_data = tapestry_data.sort_values(by = sort_by, ascending=False)

    print(tapestry_data)

    g = sns.heatmap(tapestry_data[["May", "June", "July", "August", "September"]], cmap="Blues", annot=True)

    for m in significants:
        for a in significants[m]:
            g.add_patch(Rectangle((tapestry_data.columns.get_loc(m), tapestry_data.index.tolist().index(a)), 1, 1, fill=False, edgecolor='red', lw=1))
    
    print(tapestry_data.index.tolist().index("KACV"))
    print(tapestry_data.columns.get_loc("August"))


    plt.show()