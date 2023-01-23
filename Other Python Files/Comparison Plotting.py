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

labels = pd.read_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep = "\t", names = ["acronyms", "locations"])
labels = dict(labels.values[1::])

twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "PSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY"]

def monthly_difference(airport_code):

    year_amount = 2013-1950

    month_index = []
    for year in range(1950, 1950 + 1*year_amount):
        month_index.append("May_" + str(year))
        month_index.append("June_" + str(year))
        month_index.append("July_" + str(year))
        month_index.append("August_" + str(year))
        month_index.append("September_" + str(year))

    mine = pd.read_csv('CLC_months.csv', sep = "\t")
    rachel = pd.read_csv('Rachel_months.csv', sep = "\t")

    plt.plot(month_index, mine[airport_code][:5*year_amount], color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.plot(month_index, rachel[airport_code][:5*year_amount], color='green', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.xticks(np.arange(0, 5*year_amount, 1), month_index, fontweight='bold', fontsize='6', horizontalalignment='right')

    plt.legend(["Gino", "Rachel"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

    plt.xticks(rotation = 90)

    plt.title("Gino " + airport_code + " vs Rachel " + airport_code)

    difference = pd.DataFrame(round(mine[airport_code][:5*year_amount] - rachel[airport_code][:5*year_amount], 6))

    difference['index'] = month_index

    difference.set_index('index')

    print(airport_code)
    print(np.nanmax(difference[airport_code]))
    print(np.nanmin(difference[airport_code]))

    difference.to_csv(airport_code + "_difference.csv", sep='\t')

    plt.savefig("Versus Rachel" + airport_code + ".pdf", dpi=300, format='pdf', bbox_inches='tight')

    #plt.show()

    plt.clf()

    mine = pd.read_csv('CLC_months.csv', sep = "\t")
    rachel = pd.read_csv('Rachel_months.csv', sep = "\t")

    plt.plot(month_index, rachel[airport_code][:5*year_amount], color='green', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.plot(month_index, mine[airport_code][:5*year_amount], color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

    plt.xticks(np.arange(0, 5*year_amount, 1), month_index, fontweight='bold', fontsize='6', horizontalalignment='right')

    plt.legend(["Rachel", "Gino"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

    plt.xticks(rotation = 90)

    plt.title("Gino " + airport_code + " vs Rachel " + airport_code)

    difference = pd.DataFrame(round(mine[airport_code][:5*year_amount] - rachel[airport_code][:5*year_amount], 6))

    difference['index'] = month_index

    difference.set_index('index')

    print(airport_code)
    #print(difference.to_string())
    #print(np.nanmax(difference[airport_code]))
    #print(np.nanmin(difference[airport_code]))

    difference.to_csv(airport_code + "_difference.csv", sep='\t')

    plt.savefig("Versus Gino " + airport_code + ".pdf", dpi=300, format='pdf', bbox_inches='tight')

    #plt.show()

    plt.clf()

check = 0

for file in os.listdir(r'/Users/ginoangelici/Data_Assistant_Work/Older_Files_TABLES'):
    airport_name = file[:file.index(".")]
    if airport_name in twenty_airport_acronyms:
        check += 1
        print(check)
        monthly_difference(airport_name)




