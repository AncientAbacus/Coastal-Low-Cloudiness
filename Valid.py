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
from calendar import monthrange
import warnings

def check_valid(year_frame, year, months, hours):
    valid_hours = {}
    month_number_dict = {"January": 1, "Feburary": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    for hour in hours:
        for month in months:
            # get valid times
            valid_time = year_frame[(year_frame["cld.base.ft"] != -999) & (year_frame["cld.frac"] != -999.9)]
            valid_time = valid_time[(((valid_time["cld.frac"] >= 0.75) & (valid_time["cld.base.ft"] != 72200)) | (valid_time["cld.frac"] < 0.75))]
            # filter by hour
            valid_time = valid_time[(valid_time.index.hour == hour)]
            # filter by month
            valid_time = valid_time[(valid_time.index.month == month_number_dict[month])]
            if int(valid_time[valid_time.columns[0]].count())/monthrange(year, month_number_dict[month])[1] >= 0.25:
                valid_hours[str(year) + " " + str(month) + " " + str(hour)] = True
            else:
                valid_hours[str(year) + " " + str(month) + " " + str(hour)] = False
    return all(valid_hours.values())
