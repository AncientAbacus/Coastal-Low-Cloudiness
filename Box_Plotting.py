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
from Airport_Trends import *

years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

# gets acronyms for all airports
labels = pd.read_csv('Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

# twenty aiport acronyms used for pacific rim summary + island airports KNSI and KNUC
airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

def box_and_whisker():
    box_data_1 = pd.read_csv("CLC_Data/30_Year_Eras/Avg_Tables/Airport_Monthly_CLC_Summary_Table_Years_1950_to_1979_May_June_July_August_September_Hours_7_10_13_16_.csv")
    box_data_2 = pd.read_csv("CLC_Data/30_Year_Eras/Avg_Tables/Airport_Monthly_CLC_Summary_Table_Years_1993_to_2022_May_June_July_August_September_Hours_7_10_13_16_.csv")

    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.rcParams.update({'font.size': 10})
    plt.rc('xtick', labelsize=2)
    plt.rc('ytick', labelsize=2)

    box_data_compare = pd.DataFrame()
    for airport in airport_acronyms:
        before = box_data_1[airport]
        after = box_data_2[airport]
        box_data_compare[str(airport) + "_Before"] = before
        box_data_compare[str(airport) + "_After"] = after

    # Plot the dataframe
    ax = box_data_compare.plot(kind='box', title='boxplot')
    
    plt.title(box_title)

box_title = "Box_Plot_First_vs._Last_30_Years_" + str(datetime.now().date())

box_and_whisker()

plt.savefig("Airport_Trends/Box_Plots/" + box_title + '.pdf', bbox_inches='tight')

plt.show()

"""
# Boxplot by month

years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

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

