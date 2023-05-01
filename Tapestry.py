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
from Airport_CLC_Calculation import *

# Make an image plot "tapestry" airports by latitude on y-axis, 
# months on x-axis, colors slope, mark significant. (should see nor/so Cal sig June, SoCal sig later summer signal)

#"""
# Month by Latitude graph. Monthly CLC slopes. In red if significant. 1 plot

#-------------------------------------------------------
years = range(1950, 2023)

months = ["May", "June", "July", "August", "September"]

hours = [7, 10, 13, 16]

elevation_def = 1000

y_axis = "latitude"

x_axis = months
#-------------------------------------------------------
