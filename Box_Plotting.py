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

#"""
# Box plot of first 30 CLC and last CLC by month per airport (22 graphs)

#------------------------------------------------------
years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

elevation_def = 1000

edge_length = 30
#------------------------------------------------------

def box_plotting_edge(months, hours, elevation_def, edge_length):

    details = "Months_" + listToString(months) + "_Hours_" + listToString(hours) + "_Elevation_Definition_" + str(elevation_def) 

    # gets acronyms for all airports
    labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
    labels = dict(labels.values[1::])

    # twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
    airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
    "PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]
    
    # Set the figure size
    plt.rcParams["figure.figsize"] = [10, 4]
    plt.rcParams["figure.autolayout"] = True

    plt.rcParams.update({'font.size': 6})
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)

    box_data_compare = pd.DataFrame()
    for month in months:
        box_data_1 = pd.read_csv("CLC_Data/Avg_Tables/Airport_CLC_Summary_Table_Years_1950_to_" + str(1950 + (edge_length - 1)) + "_Months_" + month + "_Hours_7_10_13_16_Elevation_Definition_1000.csv")
        box_data_2 = pd.read_csv("CLC_Data/Avg_Tables/Airport_CLC_Summary_Table_Years_" + str(2022 - (edge_length - 1)) + "_to_2022_Months_" + month + "_Hours_7_10_13_16_Elevation_Definition_1000.csv")
        for airport in airport_acronyms:
            before = box_data_1[airport]
            after = box_data_2[airport]
            box_data_compare[str(airport) + "_" + month + "\nBefore"] = before
            box_data_compare[str(airport) + "_" + month + "\nAfter"] = after

    figures = {}

    for airport in airport_acronyms:
        # Plot the dataframe
        ax = box_data_compare.filter(like=airport, axis=1).plot(kind='box', title='boxplot')
        
        box_title = "Box_Plot_First_vs._Last_" + str(edge_length) + "_Years\n" + airport +  details

        plt.title(box_title)

        figures[airport] = plt

        plt.savefig("Airport_Trends/Box_Plots/" + box_title + '.pdf', bbox_inches='tight')
    
    return figures

#"""

"""
# Boxplot by month

years = range(1950, 2023)

months = ["April"]

hours = [7, 10, 13, 16]

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

box_data_1 = pd.read_csv("CLC_Data/30_Year_Eras/Avg_Tables/Airport_Monthly_CLC_Summary_Table_Years_1950_to_1979_May_June_July_August_September_Hours_7_10_13_16_.csv")
box_data_2 = pd.read_csv("CLC_Data/30_Year_Eras/Avg_Tables/Airport_Monthly_CLC_Summary_Table_Years_1993_to_2022_May_June_July_August_September_Hours_7_10_13_16_.csv")

# Set the figure size
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.rcParams.update({'font.size': 7})

for airport in airport_acronyms:
    box_title = airport + "_Box_Plot_First_vs._Last_30_Years_May_June_July_August_September_Hours_7_10_13_16" + str(datetime.now().date())
    box_data_compare = pd.DataFrame()
    before = box_data_1[airport]
    after = box_data_2[airport]
    box_data_compare[str(airport) + "_1950_to_1979"] = before
    box_data_compare[str(airport) + "_1993_to_2022"] = after
    ax = box_data_compare.plot(kind='box', title='boxplot')
    plt.title(box_title)
    plt.ylabel("CLC (% May to Sept. \n Daytime frequency < 1000m base)")
    plt.savefig("Airport_Trends/Box_Plots/" + box_title + '.pdf', bbox_inches='tight')
    plt.clf()
"""

