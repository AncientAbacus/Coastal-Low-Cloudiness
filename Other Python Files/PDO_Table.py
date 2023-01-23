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

PDO = pd.read_csv('PDO_Data.txt',delim_whitespace=True,header=1)
PDO = PDO.loc[PDO['Year'] >= 1950]
PDO = PDO.loc[PDO['Year'] <= 2022]
PDO = PDO[["Year","May","Jun","Jul","Aug","Sep"]]
PDO['summer_PDO'] = PDO[["May","Jun","Jul","Aug","Sep"]].mean(axis=1)
PDO = PDO.reset_index()
PDO = PDO[["Year",'summer_PDO']]
PDO.to_csv('Average_Summer_PDO.csv', sep='\t')