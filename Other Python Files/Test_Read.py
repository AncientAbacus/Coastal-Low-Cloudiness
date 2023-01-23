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

ex_t = pd.read_csv('Airport_Data_from_Sam/stationdata_RESforR.txt',delim_whitespace=True,names=["c1", "c2", "c3", "c4", "c5", "c6", "c7"])

print(ex_t["c6"].loc["KACV"])
