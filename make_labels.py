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

airport_names = list(pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True).iloc[:, 1])

airport_acronyms = list(pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True).iloc[:, 0])

labels = {}
for acronym in airport_acronyms:
    for name in airport_names:
        labels[acronym] = name
        airport_names.remove(name)
        break

labels['KSIT'] = 'Sitka_Rocky_Gutierrez_Airport'
labels['PAHO'] = 'Homer_Airport'
labels['KSMX'] = 'Santa_Maria_Airport'

labels = pd.DataFrame.from_dict(labels, orient='index')
labels.to_csv(r'/Users/ginoangelici/Data_Assistant_Work/Labels.csv', sep='\t')



