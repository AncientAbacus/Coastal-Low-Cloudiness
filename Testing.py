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
import geopandas as gpd
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import Point, box
from random import uniform
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
import geopandas as gpd
from shapely.geometry import Point
from calendar import monthrange

twenty_airport_acronyms = ["PADK", "PACD", "PADQ", "PAHO", "PYAK", "KSIT", \
"PANT", "CYAZ", "KAST", "KOTH", "KACV", "KOAK", "KSFO", "KMRY", "KVBG", "KNTD", "KLAX", "KLGB", "KSAN", "KNZY", "KNSI", "KNUC"]

airport_sig_slope_counts = {}

for i in twenty_airport_acronyms:
    airport_sig_slope_counts[i] = (0, 0)

#print(airport_sig_slope_counts)

name = "Airport_Values_Summary_Table_Years_1950_to_1979_Months_May_June_July_August_September_Hours_7_10_13_16.csv"
name1 = "Airport_Values_Summary_Table_Years_1950"
name2 = "Airport_Values_Summary_Table_Years_"

print(name[35:39])
print(name[43:47])

"""
print(len(range(1994, 1994))*22)
for year in range(1950, 1994):
    loading = "["+ len(range(1950, year))*"1"+len(range(year, 1994-1))*"0"+ "]"
    print(loading)
"""

